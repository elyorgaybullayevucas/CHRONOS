#!/bin/bash
# CHRONOS uchun ma'lumotlarni bog'lash
# Agar tkg_elite2 yonida bo'lsangiz:

# Variant 1: Simlink (tavsiya etiladi)
ln -s ../tkg_elite2/data ./data
echo "Data papkasi bog'landi: $(readlink -f ./data)"

# Variant 2: Nusxalash (agar simlink ishlamasa)
# cp -r ../tkg_elite2/data ./data

# Checkpoints va logs papkalarini yaratish
mkdir -p checkpoints logs
echo "checkpoints/ va logs/ yaratildi."
echo ""
echo "Ishga tushirish:"
echo "  python main.py --dataset ICEWS18"
echo "  python main.py --dataset WIKI"
echo "  python main.py --dataset YAGO --epochs 500"
