"""SystemAwareMerger: fuses complementary Pareto frontier policies across all 3 modules."""

from __future__ import annotations

import json
from typing import Any

from src.agent.policy import ModuleName, Policy
from src.llm.client import BaseLLMClient
from src.rsi.attribution import (
    build_dialect_constraint_block,
    repair_prompt_dialect_violations,
    sanitize_rule_text,
    validate_and_sanitize_prompt,
)
from src.rsi.pareto import CandidateRecord, ParetoArchive


def find_lowest_common_ancestor(
    id_a: str,
    id_b: str,
    policies_by_id: dict[str, Policy],
) -> Policy | None:
    """Trace lineage DAG back to find a common ancestor policy."""
    def _ancestors_with_depth(start_id: str) -> dict[str, int]:
        visited: dict[str, int] = {}
        queue: list[tuple[str, int]] = [(start_id, 0)]
        while queue:
            curr, d = queue.pop(0)
            if curr in visited:
                continue
            visited[curr] = d
            pol = policies_by_id.get(curr)
            if pol:
                for pid in pol.parent_ids:
                    queue.append((pid, d + 1))
        return visited

    anc_a = _ancestors_with_depth(id_a)
    anc_b = _ancestors_with_depth(id_b)
    common = set(anc_a.keys()) & set(anc_b.keys())
    if not common:
        return None
    best_id = min(common, key=lambda cid: anc_a[cid] + anc_b[cid])
    return policies_by_id.get(best_id)


def _evolved_modules_mask(pol: Policy, seed_pol: Policy | None) -> dict[ModuleName, bool]:
    if seed_pol is None:
        return {"linker": False, "generator": False, "refiner": False}
    return {
        "linker": pol.linker_prompt.strip() != seed_pol.linker_prompt.strip(),
        "generator": pol.generator_prompt.strip() != seed_pol.generator_prompt.strip(),
        "refiner": pol.refiner_prompt.strip() != seed_pol.refiner_prompt.strip(),
    }


class SystemAwareMerger:
    """Combines non-conflicting multi-module innovations from two complementary Pareto policies."""

    @staticmethod
    def build_system_prompt(dialect: str = "odps") -> str:
        return (
            "You are SystemAwareMerger, the evolutionary crossover architect of a 3-module warehouse SQL agent "
            "(`linker`, `generator`, `refiner`).\n"
            "You are given two complementary parent policies (Candidate A and Candidate B), their validation "
            "scores by category, and their Common Ancestor.\n"
            "Synthesize a single unified child Policy (`linker_prompt`, `generator_prompt`, `refiner_prompt`) "
            "that preserves ALL non-conflicting domain rules, probing habits, dialect constraints, SCD2/anti-fanout "
            "patterns, and post-execution sanity/caliber audit checks from BOTH parents.\n\n"
            "CRITICAL CROSS-MODULE MERGE RULES:\n"
            "1. NEVER hardcode benchmark table names, task IDs, specific dates, or specific warehouse enum code literals.\n"
            "2. Preserve each module's required JSON output schema keys (`selected_tables` & `action` for linker; "
            "`thought` & `sql` for generator; `thought`, `diagnosis`, & `sql` for refiner).\n"
            "3. MODULE SPECIALIST PRESERVATION: Whenever Candidate A evolved a module (e.g., `linker` or `refiner`) "
            "relative to the Common Ancestor while Candidate B left that module at the minimal ancestor prompt, "
            "ALWAYS retain Candidate A's evolved instructions for that module rather than reverting to the minimal ancestor!\n"
            f"4. {build_dialect_constraint_block(dialect)}\n"
            "5. Deduplicate redundant wording while keeping every distinct valid rule from both parents.\n\n"
            "Output strictly valid JSON:\n"
            "{\n"
            '  "linker_prompt": "...",\n'
            '  "generator_prompt": "...",\n'
            '  "refiner_prompt": "...",\n'
            '  "merge_rationale": "..."\n'
            "}"
        )

    def __init__(self, llm: BaseLLMClient) -> None:
        self.llm = llm

    def select_complementary_pair(
        self,
        archive: ParetoArchive,
        merged_pairs_seen: set[tuple[str, str]],
        policies_by_id: dict[str, Policy] | None = None,
    ) -> tuple[CandidateRecord, CandidateRecord] | None:
        """Select the highest-complementarity unmerged pair across Pareto frontier & module specialists."""
        if len(archive.candidates) < 2:
            return None

        seed_id = archive.insertion_order[0] if archive.insertion_order else None
        seed_pol = archive.candidates[seed_id].policy if seed_id else None

        pool_ids = list(archive.get_pareto_frontier())
        # Also include the best evolved specialist per module (linker, generator, refiner) at the active depth
        # so module-orthogonal specialists can always be crossed even if scalar val_ex was slightly lower alone.
        max_depth = max(r.num_eval_tasks for r in archive.candidates.values())
        if seed_pol is not None:
            for mod in ("linker", "generator", "refiner"):
                mod_cands = [
                    r
                    for r in archive.candidates.values()
                    if r.num_eval_tasks == max_depth
                    and r.policy.get_prompt(mod).strip() != seed_pol.get_prompt(mod).strip()
                ]
                if mod_cands:
                    top_mod = max(mod_cands, key=lambda r: (r.val_ex, r.mean_score))
                    if top_mod.policy.policy_id not in pool_ids:
                        pool_ids.append(top_mod.policy.policy_id)

        if len(pool_ids) < 2:
            return None

        def _ancestors(cid: str) -> set[str]:
            if not policies_by_id:
                return {cid}
            seen_anc: set[str] = set()
            stack = [cid]
            while stack:
                cur = stack.pop()
                if cur in seen_anc:
                    continue
                seen_anc.add(cur)
                pol = policies_by_id.get(cur)
                if pol:
                    stack.extend(pol.parent_ids)
            return seen_anc

        ancestors_by_id = {cid: _ancestors(cid) for cid in archive.candidates}

        best_pair: tuple[CandidateRecord, CandidateRecord] | None = None
        best_score = -1.0

        for i in range(len(pool_ids)):
            for j in range(i + 1, len(pool_ids)):
                id_a, id_b = pool_ids[i], pool_ids[j]
                # Never waste a merge crossing the raw seed_v0 directly with its own single-step child if avoidable:
                if seed_id is not None and (id_a == seed_id or id_b == seed_id):
                    other_pol = archive.candidates[id_b if id_a == seed_id else id_a].policy
                    if other_pol.parent_ids == [seed_id]:
                        continue
                # Never cross an ancestor directly with its own descendant.
                if policies_by_id and (id_a in ancestors_by_id.get(id_b, set()) or id_b in ancestors_by_id.get(id_a, set())):
                    continue

                key = (min(id_a, id_b), max(id_a, id_b))
                if key in merged_pairs_seen:
                    continue
                rec_a = archive.candidates[id_a]
                rec_b = archive.candidates[id_b]
                cats = set(rec_a.by_category_ex.keys()) | set(rec_b.by_category_ex.keys())
                if not cats:
                    continue
                union_mean = sum(
                    max(rec_a.by_category_ex.get(c, 0.0), rec_b.by_category_ex.get(c, 0.0))
                    for c in cats
                ) / len(cats)
                max_single = max(rec_a.val_ex, rec_b.val_ex)

                # Skip redundant subset merges whose lineage is already combined by an equal/better archive candidate.
                if policies_by_id:
                    req_anc = (ancestors_by_id[id_a] | ancestors_by_id[id_b]) - ({seed_id} if seed_id else set())
                    if any(
                        other_id not in (id_a, id_b)
                        and req_anc.issubset(ancestors_by_id[other_id])
                        and archive.candidates[other_id].val_ex >= union_mean - 1e-9
                        for other_id in archive.candidates
                    ):
                        continue

                shared_tids = sorted(set(rec_a.scores_by_task.keys()) & set(rec_b.scores_by_task.keys()))
                task_gain = 0.0
                if shared_tids:
                    union_task = sum(
                        max(rec_a.scores_by_task[tid], rec_b.scores_by_task[tid]) for tid in shared_tids
                    ) / len(shared_tids)
                    task_gain = max(0.0, union_task - max(rec_a.mean_score, rec_b.mean_score))

                mask_a = _evolved_modules_mask(rec_a.policy, seed_pol)
                mask_b = _evolved_modules_mask(rec_b.policy, seed_pol)
                # Count orthogonal evolved modules brought together by crossing A and B.
                a_only_mods = sum(1 for m in ("linker", "generator", "refiner") if mask_a[m] and not mask_b[m])
                b_only_mods = sum(1 for m in ("linker", "generator", "refiner") if mask_b[m] and not mask_a[m])
                ortho_mod_bonus = (0.12 * (a_only_mods + b_only_mods)) if (a_only_mods > 0 and b_only_mods > 0) else 0.0
                union_mod_count = sum(1 for m in ("linker", "generator", "refiner") if mask_a[m] or mask_b[m])
                three_mod_bonus = 0.15 if (union_mod_count == 3 and sum(mask_a.values()) < 3 and sum(mask_b.values()) < 3) else 0.0
                mod_bonus = 0.04 if rec_a.policy.mutation_target != rec_b.policy.mutation_target else 0.0

                comp_score = (union_mean - max_single) + task_gain + ortho_mod_bonus + three_mod_bonus + mod_bonus
                if comp_score > best_score:
                    best_score = comp_score
                    best_pair = (rec_a, rec_b)

        if best_pair is not None and best_score > 0.0:
            return best_pair
        return None

    async def merge(
        self,
        rec_a: CandidateRecord,
        rec_b: CandidateRecord,
        ancestor: Policy | None,
        *,
        generation: int,
        new_policy_id: str,
        dialect: str = "odps",
    ) -> tuple[Policy, dict[str, Any]]:
        """Fuse two complementary policies via REFLECTOR_MODEL with specialist-module preservation and anti-cheat."""
        pol_a, pol_b = rec_a.policy, rec_b.policy
        anc_text = (
            f"Common Ancestor ({ancestor.policy_id}):\n"
            f"[LINKER]\n{ancestor.linker_prompt}\n"
            f"[GENERATOR]\n{ancestor.generator_prompt}\n"
            f"[REFINER]\n{ancestor.refiner_prompt}\n"
            if ancestor is not None
            else "Common Ancestor: None"
        )

        user_msg = (
            f"Target Dialect: {dialect.upper()}\n"
            f"Candidate A ({pol_a.policy_id}, val_ex={rec_a.val_ex:.3f}, "
            f"by_cat={json.dumps(rec_a.by_category_ex)}):\n"
            f"Candidate A Prompts:\n"
            f"[LINKER]\n{pol_a.linker_prompt}\n\n"
            f"[GENERATOR]\n{pol_a.generator_prompt}\n\n"
            f"[REFINER]\n{pol_a.refiner_prompt}\n\n"
            f"Candidate B ({pol_b.policy_id}, val_ex={rec_b.val_ex:.3f}, "
            f"by_cat={json.dumps(rec_b.by_category_ex)}):\n\n"
            f"Candidate B Prompts:\n"
            f"[LINKER]\n{pol_b.linker_prompt}\n\n"
            f"[GENERATOR]\n{pol_b.generator_prompt}\n\n"
            f"[REFINER]\n{pol_b.refiner_prompt}\n\n"
            f"{anc_text}\n\n"
            f"Please synthesize a merged {dialect.upper()}-compliant policy combining all non-conflicting rules "
            "across LINKER, GENERATOR, and REFINER from Candidate A and Candidate B."
        )

        resp = await self.llm.chat_json(
            [
                {"role": "system", "content": self.build_system_prompt(dialect=dialect)},
                {"role": "user", "content": user_msg},
            ],
            role_type="reflector",
            temperature=0.1,
            max_tokens=8192,
        )
        parsed = resp.parsed_json or {}
        rationale = str(parsed.get("merge_rationale", "System-aware merge of complementary Pareto parents.")).strip()
        dialect_repairs: dict[str, str] = {}

        def _preferred_parent_module(mod: ModuleName) -> str:
            pa = pol_a.get_prompt(mod).strip()
            pb = pol_b.get_prompt(mod).strip()
            if ancestor is not None:
                anc_m = ancestor.get_prompt(mod).strip()
                if pa != anc_m and pb == anc_m:
                    return pa
                if pb != anc_m and pa == anc_m:
                    return pb
            return pa if rec_a.val_ex >= rec_b.val_ex else pb

        def _validated_module(mod: ModuleName, key: str) -> str:
            cand = sanitize_rule_text(str(parsed.get(key, "")).strip(), dialect=dialect)
            # If only one parent evolved `mod` relative to ancestor and the LLM accidentally reverted to ancestor,
            # preserve the evolved parent's module prompt!
            if ancestor is not None:
                anc_m = ancestor.get_prompt(mod).strip()
                pa = pol_a.get_prompt(mod).strip()
                pb = pol_b.get_prompt(mod).strip()
                if cand == anc_m and (pa != anc_m or pb != anc_m):
                    cand = pa if pa != anc_m else pb

            ok, validated, guard_msg = validate_and_sanitize_prompt(
                cand,
                mod,
                dialect=dialect,
                auto_repair_dialect=True,
            )
            if ok:
                if guard_msg:
                    dialect_repairs[mod] = guard_msg
                return validated
            fallback_raw = _preferred_parent_module(mod)
            repaired_fb, fb_viols = repair_prompt_dialect_violations(fallback_raw, dialect=dialect)
            if fb_viols:
                dialect_repairs[mod] = f"Auto-repaired fallback {dialect.upper()} violation(s): {fb_viols}."
            return repaired_fb

        linker_p = _validated_module("linker", "linker_prompt")
        generator_p = _validated_module("generator", "generator_prompt")
        refiner_p = _validated_module("refiner", "refiner_prompt")

        if dialect_repairs:
            notes = "; ".join(f"{m}: {msg}" for m, msg in dialect_repairs.items())
            rationale = f"{rationale} [{notes}]".strip()

        merged_policy = Policy(
            policy_id=new_policy_id,
            linker_prompt=linker_p,
            generator_prompt=generator_p,
            refiner_prompt=refiner_p,
            generation=generation,
            parent_ids=[pol_a.policy_id, pol_b.policy_id],
            mutation_target="merge",
            mutation_rationale=rationale,
        )
        meta = {
            "parent_a": pol_a.policy_id,
            "parent_b": pol_b.policy_id,
            "ancestor_id": ancestor.policy_id if ancestor else None,
            "merge_rationale": rationale,
            "dialect_repairs": dialect_repairs,
        }
        return merged_policy, meta
