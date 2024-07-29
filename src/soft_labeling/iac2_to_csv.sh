#!/bin/bash
python3 iac2_to_csv.py --host localhost --user root --password '' --database createdebate --table post_view --save_path ../../data/iac2
python3 iac2_to_csv.py --host localhost --user root --password '' --database convinceme --table post_view --save_path ../../data/iac2
