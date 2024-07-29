import argparse
import pandas as pd
import os
import mysql.connector


def connect_to_mysql(host, user, password, database):
    mydb = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    mycursor = mydb.cursor()
    return mydb, mycursor


def table_to_csv(table_name, cursor, save_path):
    cursor.execute("SELECT * FROM {}".format(table_name))
    column_names = list(cursor.column_names)
    mydict = {column_names[i]:[] for i in range(len(column_names))}
    myresult = cursor.fetchall()
    for x in myresult:
        x = list(x)
        for i in range(len(x)):
            mydict[column_names[i]].append(x[i])
    df = pd.DataFrame(mydict)
    df.to_csv(save_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MySQL table to CSV')
    parser.add_argument('--host', type=str, help='MySQL host')
    parser.add_argument('--user', type=str, help='MySQL user')
    parser.add_argument('--password', type=str, help='MySQL password')
    parser.add_argument('--database', type=str, help='MySQL database')
    parser.add_argument('--table', type=str, help='MySQL table')
    parser.add_argument('--save_path', type=str, help='Path to save CSV')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    mydb, mycursor = connect_to_mysql(args.host, args.user, args.password, args.database)
    table_to_csv(args.table, mycursor, os.path.join(args.save_path, args.database + args.table + '.csv'))
    
    # test reading the csv to a dataframe and printing the first 5 rows
    #df = pd.read_csv(os.path.join(args.save_path, args.table + '.csv'))
    #print(df.head())

