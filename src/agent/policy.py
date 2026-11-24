"""Versioned multi-module agent policy triplet (linker, generator, refiner)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ModuleName = Literal["linker", "generator", "refiner"]


@dataclass
class Policy:
    """Versioned system-prompt triplet governing the 3-module warehouse SQL agent."""

    policy_id: str
    linker_prompt: str
    generator_prompt: str
    refiner_prompt: str
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutation_target: str = "seed"
    mutation_rationale: str = ""

    def get_prompt(self, module: ModuleName) -> str:
        if module == "linker":
            return self.linker_prompt
        if module == "generator":
            return self.generator_prompt
        if module == "refiner":
            return self.refiner_prompt
        raise ValueError(f"Unknown module name: {module}")

    def clone_with_mutation(
        self,
        *,
        new_policy_id: str,
        target_module: ModuleName,
        new_prompt: str,
        generation: int,
        rationale: str = "",
    ) -> "Policy":
        return Policy(
            policy_id=new_policy_id,
            linker_prompt=new_prompt if target_module == "linker" else self.linker_prompt,
            generator_prompt=new_prompt if target_module == "generator" else self.generator_prompt,
            refiner_prompt=new_prompt if target_module == "refiner" else self.refiner_prompt,
            generation=generation,
            parent_ids=[self.policy_id],
            mutation_target=target_module,
            mutation_rationale=rationale,
        )

    @classmethod
    def load_seed(cls, prompts_dir: Path, policy_id: str = "seed_v0") -> "Policy":
        linker_p = (prompts_dir / "linker_system.txt").read_text(encoding="utf-8").strip()
        generator_p = (prompts_dir / "generator_system.txt").read_text(encoding="utf-8").strip()
        refiner_p = (prompts_dir / "refiner_system.txt").read_text(encoding="utf-8").strip()
        return cls(
            policy_id=policy_id,
            linker_prompt=linker_p,
            generator_prompt=generator_p,
            refiner_prompt=refiner_p,
            generation=0,
            parent_ids=[],
            mutation_target="seed",
            mutation_rationale="Minimal generic seed prompts",
        )

    def content_hash(self, dialect: str = "") -> str:
        payload = json.dumps(
            {
                "dialect": dialect,
                "linker": self.linker_prompt.strip(),
                "generator": self.generator_prompt.strip(),
                "refiner": self.refiner_prompt.strip(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        return cls(
            policy_id=str(data["policy_id"]),
            linker_prompt=str(data["linker_prompt"]),
            generator_prompt=str(data["generator_prompt"]),
            refiner_prompt=str(data["refiner_prompt"]),
            generation=int(data.get("generation", 0)),
            parent_ids=list(data.get("parent_ids", [])),
            mutation_target=str(data.get("mutation_target", "seed")),
            mutation_rationale=str(data.get("mutation_rationale", "")),
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "Policy":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
