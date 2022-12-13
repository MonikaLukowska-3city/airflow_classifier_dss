import pyodbc
import logging
import pandas as pd

logging.basicConfig(filename='/results/output.log', level=logging.DEBUG)

class ResultsDao:
    def __init__(self, conn):
        self.conn = conn

    def save(self, model_id, df_results):
        query = '''
                INSERT INTO dss.dbo.model_results
                (
                    customer_id,
                    model_id,
                    prediction,
                    pdp_0,
                    pdp_1,
                    part
                )
                VALUES(?,?,?,?,?,?)
                '''

        logging.info(f"ResultsDao save => model_id: {model_id},  df_results: {df_results.head()}")
 
        params = []
        for row in df_results.itertuples():
            params.append( (row.customer_id, model_id, row.prediction, row.pdp_0, row.pdp_1, row.part) )

        cursor = self.conn.cursor()
        cursor.fast_executemany = True
        cursor.executemany(query, params)
        cursor.commit()


    def get_last_processed_part(self):
        query = f'''SELECT MAX(part) from dss.dbo.model_results'''
        cursor = self.conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()[0]
        cursor.close()
        return result


    def get_results_with_labels(self, model_id, data_df):
        #Szukam Label rzeczywiste ale tylko dla rezultatow ktore sa w zbiorze na ktorym byla predykcja
        key_set = set()
        for row in data_df.itertuples():
            key_set.add( (row.customer_id, row.part) )


        query = f'''
            select r.*, l.label from dss.dbo.model_results r
            join dss.dbo.new_customer_data_labels l on r.customer_id = l.customer_id and r.part = l.part
            where r.model_id = ?  
            Union
            select r.*, c.label from dss.dbo.model_results r
            join dss.dbo.customer_data c on r.customer_id = c.customer_id and r.part = c.part
            where r.model_id = ? 
        '''

        cursor = self.conn.cursor()
        
        result = []
        for row in cursor.execute(query, (model_id, model_id)):
            customer_id = row[1]
            part = row[6] 

            if (customer_id, part) in key_set:
                result.append(
                    {
                        "id": row[0],
                        "customer_id": row[1],
                        "model_id": row[2],
                        "prediction": row[3],
                        "pdp_0": row[4],
                        "pdp_1": row[5],
                        "part": row[6],
                        "label": row[7],
                    }
                )


        result_df = pd.DataFrame(result)
        return result_df
        
