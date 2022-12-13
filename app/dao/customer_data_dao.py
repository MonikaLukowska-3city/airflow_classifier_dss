import pyodbc
import pandas as pd

class CustomerDataDao:
    def __init__(self, conn):
        self.conn = conn


    def load_with_labels(self, part_id):
        query = f'''
            SELECT * FROM [dss].[dbo].[customer_data] where part <= ?
            Union
            SELECT a.part, b.label, a.customer_id, a.fea_1, a.fea_2, a.fea_3, a.fea_4, a.fea_5, a.fea_6, a.fea_7, a.fea_8, a.fea_9, a.fea_10, a.fea_11
            FROM [dss].[dbo].[new_customer_data_features] a
            join [dss].[dbo].[new_customer_data_labels] b on a.customer_id = b.customer_id and  a.part = b.part
            where a.part <= ?
        '''

        data_df = pd.read_sql(query, self.conn, params=[part_id, part_id])
        return data_df


    def load(self, part_id, use_historical = False):
        query_historical = f'''
            SELECT * FROM [dss].[dbo].[new_customer_data_features] where part <= ?
            union 
            SELECT part, customer_id, fea_1, fea_2, fea_3, fea_4, fea_5, fea_6, fea_7, fea_8, fea_9, fea_10, fea_11
            FROM [dss].[dbo].[customer_data] where part <= ?
        '''
    
        query = f'''
            SELECT * FROM [dss].[dbo].[new_customer_data_features] where part = ?
            UNION
            SELECT part, customer_id, fea_1, fea_2, fea_3, fea_4, fea_5, fea_6, fea_7, fea_8, fea_9, fea_10, fea_11
            FROM [dss].[dbo].[customer_data] where part = ?
        '''

        if use_historical:
            data_df = pd.read_sql(query_historical, self.conn, params=[part_id, part_id])
        else:
            data_df = pd.read_sql(query, self.conn, params=[part_id, part_id])

        return data_df



       

        