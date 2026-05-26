"""Domain objects for decimal arithmetic emergence probes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArithmeticExample:
    """One decimal addition equation used by the emergence probe.

    中文说明:
    - 调用方 / Called by: `ArithmeticRuleDatasetSpec`,
      `ArithmeticEmergenceService` data builders and tests.
    - 调用对象 / Calls: built-in `str`.
    - 作用 / Purpose: 用强类型封装十进制加法样例, 避免脚本层散落裸 tuple。
    - 变量 / Variables: `left/right` 是操作数, `result` 是期望和。
    - 接入 / Integration: 新增算术样例时直接构造本类型。
    - 错误处理 / Error handling: 数值合法性由数据集规约校验函数统一检查。
    - 关键词 / Keywords:
      arithmetic|addition|decimal|example|equation|prompt|answer|mhdsra2|domain|算术
    """

    left: int
    right: int
    result: int

    @property
    def prompt(self) -> str:
        """Return the equation prefix used for greedy generation.

        中文说明:
        - 调用方 / Called by: `greedy_generate_answer`, report tests.
        - 调用对象 / Calls: built-in `str` formatting.
        - 作用 / Purpose: 生成形如 `100+100=` 的输入前缀。
        - 变量 / Variables: `left/right` 来自当前样例。
        - 接入 / Integration: 生成式评估统一使用本属性。
        - 错误处理 / Error handling: 纯字符串格式化, 不吞异常。
        - 关键词 / Keywords:
          prompt|decimal|addition|generation|prefix|mhdsra2|arithmetic|domain|输入|加法
        """
        return f"{self.left}+{self.right}="

    @property
    def answer(self) -> str:
        """Return the expected generated answer string.

        中文说明:
        - 调用方 / Called by: `is_exact_generated_answer`, report renderers, tests.
        - 调用对象 / Calls: built-in `str`.
        - 作用 / Purpose: 返回不包含 `<eos>` 的十进制答案文本。
        - 变量 / Variables: `result` 是当前样例的加法结果。
        - 接入 / Integration: greedy 解码后与本属性精确比较。
        - 错误处理 / Error handling: 纯类型转换, 不吞异常。
        - 关键词 / Keywords:
          answer|decimal|addition|result|generation|mhdsra2|arithmetic|domain|答案|结果
        """
        return str(self.result)

    @property
    def equation(self) -> str:
        """Return the full equation used for teacher-forced training.

        中文说明:
        - 调用方 / Called by: `encode_training_example`, reports, tests.
        - 调用对象 / Calls: `ArithmeticExample.prompt`, `ArithmeticExample.answer`.
        - 作用 / Purpose: 生成形如 `1+1=2` 的完整训练文本。
        - 变量 / Variables: `prompt/answer` 是当前样例派生字符串。
        - 接入 / Integration: 字符级语言模型训练统一使用本属性。
        - 错误处理 / Error handling: 纯字符串拼接, 不吞异常。
        - 关键词 / Keywords:
          equation|teacher_forcing|decimal|addition|training|mhdsra2|domain|text|等式|训练
        """
        return f"{self.prompt}{self.answer}"

    @property
    def max_term(self) -> int:
        """Return the largest operand or result in this example.

        中文说明:
        - 调用方 / Called by: `ArithmeticRuleDatasetSpec.validate_training_scope`.
        - 调用对象 / Calls: built-in `max`.
        - 作用 / Purpose: 支持训练集排除百位样例的硬约束。
        - 变量 / Variables: `left/right/result` 是三个被比较的整数。
        - 接入 / Integration: 任何训练集边界检查都复用本属性。
        - 错误处理 / Error handling: 纯整数运算, 不吞异常。
        - 关键词 / Keywords:
          max_term|operand|result|scope|decimal|addition|validation|domain|百位|边界
        """
        return max(self.left, self.right, self.result)


@dataclass(frozen=True)
class ArithmeticCurriculumStage:
    """One ordered training stage in the decimal addition curriculum.

    中文说明:
    - 调用方 / Called by: `ArithmeticRuleDatasetSpec`,
      `build_curriculum_arithmetic_spec`, tests and report serialization.
    - 调用对象 / Calls: `ArithmeticExample` properties through downstream
      validators and training loops.
    - 作用 / Purpose: 用强类型表达“个位无进位 -> 进位 -> 两位数”的课程边界。
    - 变量 / Variables: `name` 是阶段名, `examples` 是该阶段的训练样例。
    - 接入 / Integration: 新增课程阶段时构造本类型并传入 `curriculum_stages`。
    - 错误处理 / Error handling: 空阶段由 `validate_training_scope` 抛出 `ValueError`。
    - 关键词 / Keywords:
      curriculum|stage|addition|decimal|ordered|training|mhdsra2|domain|课程|阶段
    """

    name: str
    examples: tuple[ArithmeticExample, ...]


@dataclass(frozen=True)
class ArithmeticRuleDatasetSpec:
    """Dataset specification for one arithmetic emergence regime.

    中文说明:
    - 调用方 / Called by: `build_default_arithmetic_spec`,
      `build_single_fact_control_spec`, `build_layer_emergence_payload`.
    - 调用对象 / Calls: `ArithmeticExample` properties.
    - 作用 / Purpose: 明确训练集、headline 外推目标、OOD 测试集和训练排除边界。
    - 变量 / Variables:
      `name` 是数据集名, `training_examples` 是低位规则样例,
      `headline_example` 是 `100+100=200`, `ood_examples` 是泛化样例。
    - 接入 / Integration: 新增实验规约时返回本类型。
    - 错误处理 / Error handling: 调用 `validate_training_scope` 暴露非法训练泄漏。
    - 关键词 / Keywords:
      dataset|spec|arithmetic|decimal|ood|headline|training|mhdsra2|domain|规约
    """

    name: str
    training_examples: tuple[ArithmeticExample, ...]
    headline_example: ArithmeticExample
    ood_examples: tuple[ArithmeticExample, ...]
    curriculum_stages: tuple[ArithmeticCurriculumStage, ...] = ()
    diagnostic_stages: tuple[ArithmeticCurriculumStage, ...] = ()
    max_training_term_exclusive: int = 100

    def validate_training_scope(self) -> None:
        """Validate that training examples do not leak held-out high-value targets.

        中文说明:
        - 调用方 / Called by: `run_arithmetic_emergence_curves`, tests.
        - 调用对象 / Calls: `ArithmeticExample.max_term`, `ArithmeticExample.equation`.
        - 作用 / Purpose: 硬性禁止训练集包含百位 operand/result、`100+100` 或 OOD 测试样例。
        - 变量 / Variables:
          `held_out_equations` 是 headline 与 OOD 等式集合,
          `training_example` 是逐个检查的训练样例。
        - 接入 / Integration: 每次运行训练前先调用本函数。
        - 错误处理 / Error handling: 发现泄漏时抛出 `ValueError`。
        - 关键词 / Keywords:
          validation|leakage|held_out|100+100|ood|training|arithmetic|decimal|scope|泄漏
        """
        if self.curriculum_stages:
            for stage in self.curriculum_stages:
                if not stage.examples:
                    raise ValueError(f"Curriculum stage is empty: {stage.name}")
            staged_examples = tuple(
                stage_example
                for stage in self.curriculum_stages
                for stage_example in stage.examples
            )
            if staged_examples != self.training_examples:
                raise ValueError(
                    "Curriculum stages must flatten to training_examples in order."
                )
        held_out_equations = {
            self.headline_example.equation,
            *[example.equation for example in self.ood_examples],
        }
        for training_example in self.training_examples:
            if training_example.max_term >= self.max_training_term_exclusive:
                raise ValueError(
                    f"Training example leaks a high-value term: {training_example.equation}"
                )
            if training_example.equation in held_out_equations:
                raise ValueError(
                    f"Training example leaks a held-out equation: {training_example.equation}"
                )


@dataclass(frozen=True)
class ArithmeticEmergenceResult:
    """Aggregated arithmetic emergence result for one model/depth/dataset.

    中文说明:
    - 调用方 / Called by: `aggregate_arithmetic_emergence_runs`,
      `find_minimum_arithmetic_emergent_layers`, report rendering.
    - 调用对象 / Calls: 无；该类型只保存聚合指标。
    - 作用 / Purpose: 表达多 seed 下某层 MHDSRA2 是否达到算术规律外推标准。
    - 变量 / Variables:
      `train_exact_match_mean/headline_exact_match_mean/ood_exact_match_mean`
      是核心成功指标, `meets_success_criteria` 是最终判定。
    - 接入 / Integration: JSON/Markdown 报告从本类型派生。
    - 错误处理 / Error handling: 聚合输入为空时由应用层抛出异常。
    - 关键词 / Keywords:
      emergence|arithmetic|aggregate|mean|variance|mhdsra2|layers|decimal|result|涌现
    """

    dataset_name: str
    model_name: str
    num_layers: int
    num_seeds: int
    train_exact_match_mean: float
    train_exact_match_variance: float
    headline_exact_match_mean: float
    headline_exact_match_variance: float
    ood_exact_match_mean: float
    ood_exact_match_variance: float
    final_loss_mean: float
    final_loss_variance: float
    meets_success_criteria: bool
