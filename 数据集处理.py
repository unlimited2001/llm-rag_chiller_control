import csv

# 假设你的 CSV 文件名为 result_total.csv，且首行为列名
# 包含列：Time, t_out, Equip, occ, light, temperature:drybulb, t_chiller, power_chiller, energy_total
# 下面脚本会在同一目录下生成 output_sentences.txt 并写入句子

with open("result_1.csv", "r", encoding="utf-8") as f_in, \
        open("output_sentences2.txt", "w", encoding="utf-8") as f_out:
    # 使用 DictReader 方便通过列名来访问数据
    reader = csv.DictReader(f_in)

    for row in reader:
        # 从 CSV 中读取各列的值
        time_str = row["Time"]
        t_out_str = row["t_out"]
        equip_str = row["Equip"]
        occ_str = row["occ"]
        light_str = row["light"]
        temp_drybulb_str = row["temperature:drybulb"]
        t_chiller_str = row["t_chiller"]
        power_chiller_str = row["power_chiller"]
        energy_total_str = row["energy_total"]

        # 按照指定格式拼接英文句子
        sentence = (
            f"\nIn the Large office, during the time {time_str}, the outdoor temperature is {t_out_str}, "
            f" the chiller's water temperature is set to {t_chiller_str} degree Celsius,"
            f"and the total energy consumption is {energy_total_str} joules.\n"

        )

        # 写入文本文件，每行一条记录
        f_out.write(sentence + "\n")

print("英文句子已写入 output_sentences2.txt 文件。")
