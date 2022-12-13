import pyodbc

class ModelsDao:
    def __init__(self, conn):
        self.conn = conn

    def save(self, model_path, ccp_alpha, numberNodes_lastTree):
        query =  '''
                SET NOCOUNT ON;
                INSERT INTO dss.dbo.models (
                    model_path,
                    ccp_alpha,
                    numberNodes_lastTree,
                    is_champion,
                    fail_count,
                    previous_acc
                )
                VALUES(?,?,?,0,0,0);
                SELECT @@IDENTITY AS id;
                '''

        cursor = self.conn.cursor()
        cursor.execute(query, (model_path, ccp_alpha, numberNodes_lastTree))
        id = cursor.fetchone()[0]
        cursor.commit()
        cursor.close()
        return int(id)


    def model_fails(self, model_id, fail):
        update_query = '''
            UPDATE dss.dbo.models
            SET fail_count = fail_count + 1                
            where id = ?;           
        '''
        fail_query = '''SELECT fail_count FROM dss.dbo.models WHERE id = ?'''

        cursor = self.conn.cursor()
        if fail:
            cursor.execute(update_query, model_id)
            cursor.commit()

        cursor.execute(fail_query, model_id)
        fail_count = cursor.fetchone()[0]
        cursor.close()
        
        return fail_count



    def save_score(self, model_id, precision_score, recall_score, roc_auc_score, acc_val):
        query = '''
            UPDATE dss.dbo.models
            SET precision_score = ?,
                recall_score = ?,
                roc_auc_score = ?,
                previous_acc = acc,
                acc = ?
            where id = ?
        '''
        cursor = self.conn.cursor()
        cursor.execute(query, (precision_score, recall_score, roc_auc_score, acc_val, model_id))
        cursor.commit()
        cursor.close()


    def get_champion_and_challenger_models(self):
        query = "select id, challenger_id from dss.dbo.models where is_champion = 1"
        cursor = self.conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()

        if result is not None:
            return result[0], result[1]
        return None, None 



    def set_challenger_model(self, champion_id, challenger_id):
        query = '''
            UPDATE dss.dbo.models set challenger_id = ? where id = ?;            
            '''
        cursor = self.conn.cursor()
        cursor.execute(query, (challenger_id, champion_id))
        cursor.commit()
        cursor.close()

    
    def get_challenger_model(self, model_id):
        query = 'select challenger_id from dss.dbo.models where id = ?'
        cursor = self.conn.cursor()
        cursor.execute(query, model_id)
        result = cursor.fetchone()[0]
        cursor.close()
        return result

        

    def set_champion_model(self, model_id):
        query = '''
            UPDATE dss.dbo.models set is_champion = 0 where id != ?;
            UPDATE dss.dbo.models set is_champion = 1 where id = ?;
            '''
        cursor = self.conn.cursor()
        cursor.execute(query, (model_id, model_id))
        cursor.commit()
        cursor.close()


    def get_champion_model_path_and_id(self):
        query = 'select model_path, id from dss.dbo.models where is_champion = 1'
        cursor = self.conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()

        if result is None:
            return None, None

        return result[0], result[1]


    def get_model_path(self, model_id):
        query = 'select model_path from dss.dbo.models where id = ?'
        cursor = self.conn.cursor()
        cursor.execute(query, model_id)
        path = cursor.fetchone()[0]
        cursor.close()
        return path

 
    def get_model_acc(self, model_id):
        query = 'select acc, previous_acc from dss.dbo.models where id = ?'
        cursor = self.conn.cursor()
        cursor.execute(query, model_id)
        result = cursor.fetchone()
        cursor.close()

        if result is not None:
            return result[0], result[1]
        return None, None

    def get_model_roc_auc(self, model_id):
        query = 'select roc_auc_score from dss.dbo.models where id = ?'
        cursor = self.conn.cursor()
        cursor.execute(query, model_id)
        roc_auc_score = cursor.fetchone()[0]
        cursor.close()
        return roc_auc_score