import pyodbc

class PdpDecilesDao:
    def __init__(self, conn):
        self.conn = conn

    def save(self, model_id, deciles):
        query = '''
                SET NOCOUNT ON;
                INSERT INTO dss.dbo.model_pdp_deciles(
                    model_id,
                    deciles_1,
                    deciles_2,
                    deciles_3,
                    deciles_4,
                    deciles_5,
                    deciles_6,
                    deciles_7,
                    deciles_8,
                    deciles_9,
                    deciles_10
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?);
                SELECT @@IDENTITY AS id;
                '''

        params = [model_id]
        params.extend(deciles)                

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        id = cursor.fetchone()[0]
        cursor.commit()
        cursor.close()
        return int(id)