import pyodbc

class MetadataDao:
    def __init__(self, conn):
        self.conn = conn

    def save(self, model_id, features):
        query = '''
                SET NOCOUNT ON;
                INSERT INTO dss.dbo.model_metadata(
                    model_id,
                    fi_fea_1,
                    fi_fea_2,
                    fi_fea_3,
                    fi_fea_4,
                    fi_fea_5,
                    fi_fea_6,
                    fi_fea_7,
                    fi_fea_8,
                    fi_fea_9,
                    fi_fea_10,
                    fi_fea_11
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?);
                SELECT @@IDENTITY AS id;
                '''

        params = [model_id]
        params.extend(features) 

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        id = cursor.fetchone()[0]
        cursor.commit()
        cursor.close()
        return int(id)