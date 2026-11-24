"""Enterprise 15-table data warehouse catalog definition (v1 core + Hard-v2 SCD2/settlement schemas)."""

from __future__ import annotations

from src.env.schema import ColumnMeta, TableMeta, WarehouseCatalog


def build_warehouse_catalog() -> WarehouseCatalog:
    """Define the 15-table enterprise e-commerce & content data warehouse catalog."""
    catalog = WarehouseCatalog()

    catalog.register(
        TableMeta(
            name="dim_user",
            layer="dim",
            comment="用户基础维表（非分区全量维表）",
            partition_col=None,
            columns=(
                ColumnMeta("user_id", "VARCHAR", "用户唯一ID"),
                ColumnMeta("reg_dt", "VARCHAR", "注册日期 YYYY-MM-DD"),
                ColumnMeta("vip_level", "VARCHAR", "会员等级: normal/silver/gold/platinum"),
                ColumnMeta("city_tier", "VARCHAR", "城市线级: T1/T2/T3，存在NULL和空字符串"),
                ColumnMeta("gender", "VARCHAR", "性别: M/F，存在NULL和空字符串"),
                ColumnMeta("channel", "VARCHAR", "拉新渠道: organic/paid_ads/referral/live_stream"),
                ColumnMeta("is_blacklist", "INTEGER", "是否黑产风控用户: 0=正常, 1=黑名单"),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dim_shop",
            layer="dim",
            comment="商家店铺维表（非分区全量维表）",
            partition_col=None,
            columns=(
                ColumnMeta("shop_id", "VARCHAR", "店铺唯一ID"),
                ColumnMeta("shop_name", "VARCHAR", "店铺名称"),
                ColumnMeta("shop_type", "VARCHAR", "店铺类型: flagship/pop_store/cross_border/brand_direct"),
                ColumnMeta("main_cate", "VARCHAR", "主营一级类目"),
                ColumnMeta("open_dt", "VARCHAR", "开店日期 YYYY-MM-DD"),
                ColumnMeta("rating_score", "DOUBLE", "店铺综合评分"),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dim_item",
            layer="dim",
            comment="商品SKU维表（非分区全量维表）",
            partition_col=None,
            columns=(
                ColumnMeta("item_id", "VARCHAR", "商品ID"),
                ColumnMeta("shop_id", "VARCHAR", "所属店铺ID"),
                ColumnMeta("cate_l1", "VARCHAR", "一级类目"),
                ColumnMeta("cate_l2", "VARCHAR", "二级类目"),
                ColumnMeta("brand", "VARCHAR", "品牌名称，存在NULL和空字符串"),
                ColumnMeta("list_price", "DOUBLE", "商品吊牌价"),
                ColumnMeta("status", "VARCHAR", "商品状态: on_sale/off_shelf"),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dim_content_creator",
            layer="dim",
            comment="内容达人维表（非分区全量维表）",
            partition_col=None,
            columns=(
                ColumnMeta("creator_id", "VARCHAR", "达人ID"),
                ColumnMeta("creator_name", "VARCHAR", "达人昵称"),
                ColumnMeta("creator_tier", "VARCHAR", "达人层级: head/waist/tail"),
                ColumnMeta("mcn_org", "VARCHAR", "签约MCN机构，存在NULL和空字符串"),
                ColumnMeta("fans_cnt", "INTEGER", "粉丝数"),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="ods_app_event_log_di",
            layer="ods",
            comment="App端用户行为埋点增量日志表（含重复上报、JSON扩展字段与曝光商品数组）",
            partition_col="ds",
            columns=(
                ColumnMeta("event_id", "VARCHAR", "埋点事件ID（存在跨端/重试重复上报）"),
                ColumnMeta("user_id", "VARCHAR", "用户ID"),
                ColumnMeta("event_type", "VARCHAR", "事件类型: exposure/click/add_cart/purchase"),
                ColumnMeta("page_name", "VARCHAR", "页面名称: home_feed/search_result/live_room/item_detail"),
                ColumnMeta("ingest_ts", "VARCHAR", "服务端落库时间 YYYY-MM-DD HH:MM:SS（去重取最新）"),
                ColumnMeta("ext_json", "VARCHAR", "扩展JSON字符串，含$.page, $.stay_ms, $.ab_group, $.device.os，存在NULL/空串"),
                ColumnMeta("expo_item_ids", "VARCHAR[]", "曝光商品ID列表，部分事件为空数组[]"),
                ColumnMeta("ds", "VARCHAR", "日分区 YYYYMMDD", is_partition=True),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dwd_trade_order_di",
            layer="dwd",
            comment="交易订单明细增量事实表（子订单粒度，同一order_id可能包含多条商品明细）",
            partition_col="ds",
            columns=(
                ColumnMeta("order_id", "VARCHAR", "主订单ID（一单多品时对应多行）"),
                ColumnMeta("order_item_id", "VARCHAR", "子订单明细行ID"),
                ColumnMeta("user_id", "VARCHAR", "下单用户ID"),
                ColumnMeta("shop_id", "VARCHAR", "店铺ID"),
                ColumnMeta("item_id", "VARCHAR", "商品ID"),
                ColumnMeta("creator_id", "VARCHAR", "带货达人ID，非达人订单为NULL或空字符串"),
                ColumnMeta("order_status", "VARCHAR", "订单状态: paid/shipped/completed/cancelled"),
                ColumnMeta("qty", "INTEGER", "购买件数"),
                ColumnMeta("gross_amt", "DOUBLE", "应付原价金额"),
                ColumnMeta("discount_amt", "DOUBLE", "优惠金额，无优惠时为NULL"),
                ColumnMeta("pay_amt", "DOUBLE", "实付金额（取消单为0）"),
                ColumnMeta("pay_ts", "VARCHAR", "支付时间 YYYY-MM-DD HH:MM:SS，未支付为NULL"),
                ColumnMeta("ds", "VARCHAR", "交易日分区 YYYYMMDD", is_partition=True),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dwd_refund_di",
            layer="dwd",
            comment="售后退款明细增量事实表（同一order_id可能存在多次部分退款）",
            partition_col="ds",
            columns=(
                ColumnMeta("refund_id", "VARCHAR", "退款单ID"),
                ColumnMeta("order_id", "VARCHAR", "关联主订单ID"),
                ColumnMeta("user_id", "VARCHAR", "用户ID"),
                ColumnMeta("shop_id", "VARCHAR", "店铺ID"),
                ColumnMeta("refund_amt", "DOUBLE", "退款金额"),
                ColumnMeta("refund_reason", "VARCHAR", "退款原因: quality/size_mismatch/late_delivery/buyer_remorse"),
                ColumnMeta("refund_status", "VARCHAR", "退款状态: approved/pending/rejected"),
                ColumnMeta("ds", "VARCHAR", "退款日分区 YYYYMMDD", is_partition=True),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dwd_content_post_di",
            layer="dwd",
            comment="内容种草帖子增量明细表（注意分区字段为dt而非ds）",
            partition_col="dt",
            columns=(
                ColumnMeta("post_id", "VARCHAR", "内容帖子ID"),
                ColumnMeta("creator_id", "VARCHAR", "达人ID"),
                ColumnMeta("content_type", "VARCHAR", "内容形式: short_video/live_replay/article"),
                ColumnMeta("topic_tags", "VARCHAR[]", "话题标签数组"),
                ColumnMeta("meta_json", "VARCHAR", "内容元数据JSON，含$.duration_sec, $.is_promoted, $.score"),
                ColumnMeta("view_cnt", "INTEGER", "曝光浏览量"),
                ColumnMeta("like_cnt", "INTEGER", "点赞量"),
                ColumnMeta("dt", "VARCHAR", "内容发布日分区 YYYYMMDD", is_partition=True),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dws_shop_metrics_1d",
            layer="dws",
            comment="店铺日度流量与交易汇总表（含uv=0、ad_spend=0/NULL等冷启动/空值场景）",
            partition_col="ds",
            columns=(
                ColumnMeta("shop_id", "VARCHAR", "店铺ID"),
                ColumnMeta("uv", "INTEGER", "日访客数"),
                ColumnMeta("pv", "INTEGER", "日浏览量"),
                ColumnMeta("order_cnt", "INTEGER", "日有效订单数"),
                ColumnMeta("buyer_cnt", "INTEGER", "日成交买家数"),
                ColumnMeta("gmv", "DOUBLE", "日成交GMV"),
                ColumnMeta("return_amt", "DOUBLE", "日退货金额，无退货时可能为NULL"),
                ColumnMeta("ad_spend", "DOUBLE", "日广告投放花费，无投放时为0或NULL"),
                ColumnMeta("ds", "VARCHAR", "日分区 YYYYMMDD", is_partition=True),
            ),
        )
    )

    # --- Hard-v2 Enterprise Deep-Water Tables ---
    catalog.register(
        TableMeta(
            name="dwd_trade_order_legacy_di",
            layer="dwd",
            comment="交易订单结算快照增量表（V1历史链路）",
            partition_col="ds",
            columns=(
                ColumnMeta("order_id", "VARCHAR", "订单ID"),
                ColumnMeta("user_id", "VARCHAR", "用户ID"),
                ColumnMeta("shop_id", "VARCHAR", "店铺ID"),
                ColumnMeta("pay_channel", "VARCHAR", "支付渠道"),
                ColumnMeta("settle_status", "VARCHAR", "结算状态"),
                ColumnMeta("order_amt", "DOUBLE", "订单结算金额"),
                ColumnMeta("ds", "VARCHAR", "日分区", is_partition=True),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dwd_order_settle_di",
            layer="dwd",
            comment="全端订单财务清分结算增量表（企业V2主单标准明细）",
            partition_col="ds",
            columns=(
                ColumnMeta("settle_id", "VARCHAR", "清分单ID"),
                ColumnMeta("order_id", "VARCHAR", "主订单ID"),
                ColumnMeta("user_id", "VARCHAR", "买家用户ID"),
                ColumnMeta("shop_id", "VARCHAR", "店铺ID"),
                ColumnMeta("order_dt", "VARCHAR", "实际下单日期"),
                ColumnMeta("pay_channel_code", "VARCHAR", "支付渠道编码"),
                ColumnMeta("settle_status_code", "VARCHAR", "清分状态编码"),
                ColumnMeta("gross_order_amt_cent", "BIGINT", "订单应付毛额(分)"),
                ColumnMeta("settle_amt_cent", "BIGINT", "渠道实收清分额(分)"),
                ColumnMeta("platform_subsidy_cent", "BIGINT", "平台补贴额(分)"),
                ColumnMeta("merchant_net_amt", "BIGINT", "商家实得清分净额"),
                ColumnMeta("is_test_acct", "INTEGER", "内测压测账号标识"),
                ColumnMeta("ds", "VARCHAR", "财务清分日分区", is_partition=True),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dwd_aftersale_refund_di",
            layer="dwd",
            comment="售后逆向退赔单增量明细表",
            partition_col="dt",
            columns=(
                ColumnMeta("aftersale_id", "VARCHAR", "售后退赔单ID"),
                ColumnMeta("order_id", "VARCHAR", "关联主订单ID"),
                ColumnMeta("user_id", "VARCHAR", "买家用户ID"),
                ColumnMeta("shop_id", "VARCHAR", "店铺ID"),
                ColumnMeta("refund_type_code", "VARCHAR", "售后退赔类型编码"),
                ColumnMeta("refund_status_code", "VARCHAR", "退赔状态编码"),
                ColumnMeta("refund_amt_cent", "BIGINT", "退赔货款额"),
                ColumnMeta("shipping_comp_yuan", "DOUBLE", "保司运费险赔付额"),
                ColumnMeta("dt", "VARCHAR", "分区日期", is_partition=True),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dim_user_vip_scd2",
            layer="dim",
            comment="用户会员等级变更拉链表",
            partition_col=None,
            columns=(
                ColumnMeta("scd_id", "VARCHAR", "拉链记录ID"),
                ColumnMeta("user_id", "VARCHAR", "用户ID"),
                ColumnMeta("vip_tier", "VARCHAR", "会员等级"),
                ColumnMeta("growth_score", "INTEGER", "版本期初成长值"),
                ColumnMeta("eff_start_dt", "VARCHAR", "版本生效开始日"),
                ColumnMeta("eff_end_dt", "VARCHAR", "版本生效结束日"),
                ColumnMeta("is_current", "INTEGER", "当前有效标识"),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dim_shop_commission_scd2",
            layer="dim",
            comment="店铺签约抽佣费率拉链表",
            partition_col=None,
            columns=(
                ColumnMeta("comm_scd_id", "VARCHAR", "费率版本ID"),
                ColumnMeta("shop_id", "VARCHAR", "店铺ID"),
                ColumnMeta("contract_tier", "VARCHAR", "商业化签约档位"),
                ColumnMeta("comm_rate_bps", "INTEGER", "技术服务费率(bps)"),
                ColumnMeta("eff_start_dt", "VARCHAR", "生效开始日"),
                ColumnMeta("eff_end_dt", "VARCHAR", "生效结束日"),
                ColumnMeta("is_current", "INTEGER", "当前有效标识"),
            ),
        )
    )

    catalog.register(
        TableMeta(
            name="dws_shop_operating_1d",
            layer="dws",
            comment="店铺日度经营与风控口径汇总表",
            partition_col="ds",
            columns=(
                ColumnMeta("shop_id", "VARCHAR", "店铺ID"),
                ColumnMeta("biz_date", "VARCHAR", "业务日期"),
                ColumnMeta("op_status_code", "VARCHAR", "当日营业状态编码"),
                ColumnMeta("abnormal_flag", "VARCHAR", "风控状态标记"),
                ColumnMeta("gross_gmv_yuan", "DOUBLE", "账面毛成交额"),
                ColumnMeta("clean_gmv_yuan", "DOUBLE", "有效净成交额"),
                ColumnMeta("test_gmv_yuan", "DOUBLE", "内测成交额"),
                ColumnMeta("promo_cost_yuan", "DOUBLE", "营销推广消耗"),
                ColumnMeta("valid_order_cnt", "INTEGER", "有效订单数"),
                ColumnMeta("ds", "VARCHAR", "日分区", is_partition=True),
            ),
        )
    )

    return catalog
