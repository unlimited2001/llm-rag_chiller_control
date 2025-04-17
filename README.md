# llm-rag_chiller_control
eee dissertation(llm-rag hvac chiller control)
llm-rag全流程：
1.运行test_enhanced2.py，会生成6-12度冷机出水温度数据，已放在result_1.csv
2.运行"数据集处理.py"，将数据整理成可以分割的数据集文本，已放在output_sentences2.txt
3.运行model-openai-rag.py，调用rag输出llm-rag方案的冷机出水温度。
4.运行"处理llm输出.py"，将llm-rag方案的冷机出水温度整理为一段可以输入进energyplus的时间序列
5.运行"与energyplus交互.py"，得到llm-rag方案的仿真结果
llm全流程：
1.运行model.py，调用rag输出llm方案的冷机出水温度。
4.运行"处理llm输出.py"，将llm方案的冷机出水温度整理为一段可以输入进energyplus的时间序列
5.运行"与energyplus交互.py"，得到llm方案的仿真结果
