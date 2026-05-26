"""Domain model specifications for retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass


ARCHIVED_MODEL_ALIASES: dict[str, str] = {
    "dsra": "mhdsra2",
}


def normalize_model_type(model_type: str) -> str:
    """Normalize archived model names to the active architecture.

    中文说明:
    - 调用方 / Called by: `RetrievalModelSpec.__post_init__`, JSON retrieval builders
    - 调用对象 / Calls: `str.lower`, dictionary lookup
    - 作用 / Purpose: 将已归档的 `dsra` 架构名统一映射到当前主架构 `mhdsra2`
    - 变量 / Variables:
      `model_type` 是外部输入模型名, `normalized` 是标准化小写值
    - 接入 / Integration: 所有模型构建入口先调用本函数再分派构造器
    - 错误处理 / Error handling: 空模型名抛出 `ValueError`
    - 关键词 / Keywords:
      normalize|model_type|archive|dsra|mhdsra2|alias|domain|retrieval|factory|归档
    """
    normalized = model_type.strip().lower()
    if not normalized:
        raise ValueError("model_type must not be empty")
    return ARCHIVED_MODEL_ALIASES.get(normalized, normalized)


@dataclass(frozen=True)
class RetrievalModelSpec:
    """Describe a retrieval benchmark model construction request.

    中文说明:
    - 调用方 / Called by: `scripts.json_retrieval_test.build_retrieval_model`
    - 调用对象 / Calls: `normalize_model_type`, `ValueError`
    - 作用 / Purpose: 把 JSON retrieval 模型构建参数从脚本层提升为领域规格
    - 变量 / Variables:
      `model_type` 是归一化后模型族, `requested_model_type` 保留用户输入,
      `vocab_size/dim/slots/topk/chunk_size` 是模型结构参数,
      `local_context_size/local_context_mode` 控制局部上下文编码
    - 接入 / Integration: 传给 `RetrievalModelFactory.build`
    - 错误处理 / Error handling: 非法数值和局部上下文模式抛出 `ValueError`
    - 关键词 / Keywords:
      retrieval|model_spec|domain|factory|mhdsra2|json|local_context|config|slots|模型规格
    """

    requested_model_type: str
    vocab_size: int
    dim: int
    slots: int
    topk: int
    chunk_size: int
    local_context_size: int
    local_context_mode: str

    @property
    def model_type(self) -> str:
        """Return the active architecture name after archive alias normalization.

        中文说明:
        - 调用方 / Called by: `RetrievalModelFactory.build`, tests and configs
        - 调用对象 / Calls: `normalize_model_type`
        - 作用 / Purpose: 暴露当前实际使用的模型类型
        - 变量 / Variables: `requested_model_type` 是用户或脚本传入值
        - 接入 / Integration: 构造器注册表使用该属性分派
        - 错误处理 / Error handling: 透传 `normalize_model_type` 的空值错误
        - 关键词 / Keywords:
          model_type|normalized|archive|mhdsra2|dsra|alias|property|factory|domain|归一化
        """
        return normalize_model_type(self.requested_model_type)

    def __post_init__(self) -> None:
        """Validate retrieval model construction parameters.

        中文说明:
        - 调用方 / Called by: `dataclasses` 初始化后自动调用
        - 调用对象 / Calls: `normalize_model_type`, `ValueError`
        - 作用 / Purpose: 在模型构造前拦截非法领域参数
        - 变量 / Variables:
          `local_context_modes` 是允许的上下文模式集合，其余字段来自当前规格对象
        - 接入 / Integration: 新增模型结构参数时在此补充领域约束
        - 错误处理 / Error handling: 非法参数抛出 `ValueError`
        - 关键词 / Keywords:
          validate|model_spec|retrieval|domain|local_context|dim|slots|topk|chunk|校验
        """
        _ = self.model_type
        local_context_modes = {"sum", "concat", "none"}
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.dim < 1:
            raise ValueError("dim must be positive")
        if self.slots < 1:
            raise ValueError("slots must be positive")
        if self.topk < 1:
            raise ValueError("topk must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.local_context_size < 1:
            raise ValueError("local_context_size must be positive")
        if self.local_context_mode not in local_context_modes:
            raise ValueError(f"Unsupported local_context_mode: {self.local_context_mode}")
