from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()

input_work = input("創作のインプットにしたい作品名を入力してください: ")
plot_conditions = []

while True:
    plot_condition = input(
        "プロットを生成した作品の設定を好きなだけ入力してください (終了するには 'exit' と入力): "
    )
    if plot_condition.lower() == "exit":
        break
    else:
        plot_conditions.append(plot_condition)

plot_conditions_md = ""
for i, condition in enumerate(plot_conditions, start=1):
    plot_conditions_md += f"{i}. {condition}\n"

print("以下の条件で「", input_work, "」に似た物語構造のプロットを生成します。", sep="")
print(plot_conditions_md)


examples = [
    {
        "work": "桃太郎",
        "plot": """
1.社会や共同体が危機に直面している。
2.特別な力や資質を持つ主人公が登場し、問題解決の鍵を握る。
3.主人公が課題を認識し、目標に向かって行動することを決意する。
4.主人公が目的達成に必要なリソースや助力を集める。
5.主人公が敵対者や障害に直面し、それを乗り越える。
6.主人公が課題を達成し、成果や報酬を得る。
7.主人公が成果を持ち帰り、社会に還元することで秩序が回復する。
        """,
    },
    {
        "work": "シンデレラ",
        "plot": """
1.主人公が困難や不公正な状況に置かれている。
2.主人公が新たなチャンスや希望を見つけるが、現状に阻まれる。
3.主人公が課題を克服するための助力者やリソースを得る。
4.主人公が試練を受け入れ、一時的な成功を収めるが、問題が完全には解決しない。
5.主人公の成功が危機にさらされるが、それを解決するための手がかりが提示される。
6.主人公が真の価値や能力を認識され、課題が完全に解決される。
7.主人公が自己実現を果たし、物語世界の秩序が回復または再構築される。
        """,
    },
]

example_prompt = PromptTemplate(
    input_variables=["work", "plot"], template="# 「{work}」の場合\n{plot}"
)


class PlotResult(BaseModel):
    episodes: list[str] = Field(description="プロットを構成するエピソードのリスト")


parser = JsonOutputParser(pydantic_object=PlotResult)

get_input_work_plot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="以下の例のように「{input_work}」の物語構造を抽象化してください。",
    suffix="{format_instructions}",
    input_variables=["input_work"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

get_input_work_plot_prompt = get_input_work_plot_prompt_template.format(
    input_work=input_work
)

model = ChatOpenAI(model="gpt-4o")
output = model.invoke(get_input_work_plot_prompt)
result = parser.invoke(output)

if "episodes" not in result:
    print(output)
    exit()

"""
`result`の中身は以下のようになる。
{
	"episodes": [
		"主人公が現実世界での孤独や疎外感を抱えている。",
		"主人公が異世界への旅に出ることを決意する。",
		"異世界での旅の中で、主人公が様々な出会いや体験を通じて自己理解を深める。",
		"主人公が旅を通じて、人間関係や生命についての深い洞察を得る。",
		"旅の終わりが近づく中で、主人公が選択を迫られ、精神的な成長を遂げる。",
		"主人公が元の世界に帰還し、異世界での経験を通じて現実世界での生き方を見直す。",
		"物語を通じて、主人公が自己の内面と外界の調和を見出し、心の平穏を得る。"
	]
}
"""

get_your_plot_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("""
以下のプロットで物語を生成してください。ただし、与えた設定を含めてください。

# プロット
{plot}
"""),
        HumanMessage(content=plot_conditions_md),
    ]
)

input_plot = ""
for i, event in enumerate(result["episodes"], start=1):
    input_plot += f"{i}. {event}\n"

get_your_plot_prompt = get_your_plot_prompt_template.format(plot=input_plot)

print(get_your_plot_prompt)
output = model.invoke(get_your_plot_prompt)
print(output)
