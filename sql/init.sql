CREATE DATABASE dss;
GO

CREATE TABLE dss.dbo.customer_data (
    part int,
    label int,
    customer_id int,
    fea_1 int,
    fea_2 float,
    fea_3 int,
    fea_4 float,
    fea_5 int,
    fea_6 int,
    fea_7 int,
    fea_8 int,
    fea_9 int,
    fea_10 int,
    fea_11 float
);
GO

CREATE TABLE dss.dbo.new_customer_data_features (
    part int,
    customer_id int,
    fea_1 int,
    fea_2 float,
    fea_3 int,
    fea_4 float,
    fea_5 int,
    fea_6 int,
    fea_7 int,
    fea_8 int,
    fea_9 int,
    fea_10 int,
    fea_11 float
);
GO

CREATE TABLE dss.dbo.new_customer_data_labels (
    part int,
    label int,
    customer_id int
);
GO

CREATE TABLE dss.dbo.models (
    id int IDENTITY(1,1) PRIMARY KEY,
    model_path text,
    acc float,
    previous_acc float,
    precision_score float,
    recall_score float,
    roc_auc_score float,
    ccp_alpha float,
    numberNodes_lastTree int,
    is_champion int,
    fail_count int,
    challenger_id int
);
GO


CREATE TABLE dss.dbo.model_metadata (
    id int IDENTITY(1,1) PRIMARY KEY,
    model_id float,
    fi_fea_1 float,
    fi_fea_2 float,
    fi_fea_3 float,
    fi_fea_4 float,
    fi_fea_5 float,
    fi_fea_6 float,
    fi_fea_7 float,
    fi_fea_8 float,
    fi_fea_9 float,
    fi_fea_10 float,
    fi_fea_11 float
);
GO

CREATE TABLE dss.dbo.model_pdp_deciles (
    id int IDENTITY(1,1) PRIMARY KEY,
    model_id float,
    deciles_1 float,
    deciles_2 float,
    deciles_3 float,
    deciles_4 float,
    deciles_5 float,
    deciles_6 float,
    deciles_7 float,
    deciles_8 float,
    deciles_9 float,
    deciles_10 float
);
GO

CREATE TABLE dss.dbo.model_results (
    id int IDENTITY(1,1) PRIMARY KEY,
    customer_id int,
    model_id float,
    prediction int,
    pdp_0 float,
    pdp_1 float,
    part int
);
GO

-- IMPORT THE FILE
BULK INSERT dss.dbo.customer_data
FROM
'/tmp/mssql/customer_data.csv'
WITH
(
    FORMAT='CSV',
    FIRSTROW=2
)
GO


-- IMPORT THE FILE
BULK INSERT dss.dbo.new_customer_data_features
FROM
'/tmp/mssql/new_customer_data_features.csv'
WITH
(
    FORMAT='CSV',
    FIRSTROW=2
)
GO

-- IMPORT THE FILE
BULK INSERT dss.dbo.new_customer_data_labels
FROM
'/tmp/mssql/new_customer_data_labels.csv'
WITH
(
    FORMAT='CSV',
    FIRSTROW=2
)
GO