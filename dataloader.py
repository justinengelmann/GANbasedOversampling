import pandas as pd
import numpy as np

import logging


def get_datasets(names_only: bool = False):
    DATASET_DICT = {
        'statlog': load_statlog,
        'bank': load_bank,
        'thomas': load_thomas,
        'pakdd': load_pakdd,
        'taiwan': load_taiwan,
        'homeeq': load_homeeq,
        'lendingcluba': load_lendingcluba,
        'lendingclubb': load_lendingclubb,
        'gmc': load_gmc,
        'dmc_05': load_dmc05,
        'dmc_10': load_dmc10,
        'coil2k': load_coil2k,
        'adult': load_adult,
        'statlog_australian': load_statlog_australian
    }

    if names_only:
        return list(DATASET_DICT.keys())
    else:
        return DATASET_DICT


def get_dataset_setting(dataset: str) -> str:
    settings = {'statlog': 'Credit scoring',
                'bank': 'Marketing',
                'homeeq': 'Credit scoring',
                'dmc_05': 'Profitability scoring',
                'dmc_10': 'Response modeling',
                'coil2k': 'Response modeling',
                'adult': 'Income prediction'}
    setting = settings[dataset]
    return setting


def get_dataset_source(dataset: str) -> str:
    sources = {'statlog': 'UCI MLR',
               'bank': 'UCI MLR',
               'homeeq': 'Baesens et al.',
               'dmc_05': 'DMC 2005',
               'dmc_10': 'DMC 2010',
               'coil2k': 'UCI MLR',
               'adult': 'UCI MLR'}
    source = sources[dataset]
    return source


def load_data(dataset: str):
    logging.debug(f'Dataloader: Loading {dataset}')

    dataset_dict = get_datasets()

    logging.debug(f'Dataloader: Loaded available datasets.')

    if dataset in dataset_dict.keys():
        func = dataset_dict[dataset]
        df, cat_cols, num_cols, target_col = func()
    else:
        logging.error(f'Dataloader: Dataset {dataset} not found.')
        raise ValueError(f'Dataloader: Dataset "{dataset}" not found.')

    df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes.astype('category'))

    logging.info(f'Dataloader: Loaded dataset: {dataset}. Returning data.')

    return df, cat_cols, num_cols, target_col


# ## Preprocessing

# #### statlog
# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

def load_statlog():
    path = 'Datasets/Raw/UCI_statlog_german_credit_data_data_set/german.data'

    col_names = ['Status_checking_account', 'Duration_months', 'Credit_history', 'Purpose',
                 'Credit_amount', 'Savings_account_bonds', 'Present_employment_since',
                 'Instalment_rate_percent_of_income', 'Personal_status_sex', 'Other_debtors_guarantors',
                 'Present_residence_since', 'Property', 'Age_years', 'Other_instalment_plans',
                 'Housing', 'Number_of_existing_credits', 'Job', 'Dependants', 'Telephone',
                 'Foreign_worker', 'Status_loan']

    cat_cols = ['Status_checking_account', 'Credit_history', 'Purpose',
                'Savings_account_bonds', 'Present_employment_since', 'Personal_status_sex',
                'Other_debtors_guarantors', 'Property', 'Other_instalment_plans', 'Housing',
                'Job', 'Telephone', 'Foreign_worker']
    num_cols = ['Duration_months', 'Credit_amount', 'Instalment_rate_percent_of_income',
                'Present_residence_since', 'Age_years', 'Number_of_existing_credits', 'Dependants']

    target_col = 'Status_loan'

    df = pd.read_csv(path, sep=' ', header=None, index_col=False,
                     names=col_names,
                     dtype={col: 'category' for col in cat_cols})

    df[target_col] = df[target_col] - 1

    return df, cat_cols, num_cols, target_col


# #### statlog australian
# http://archive.ics.uci.edu/ml/datasets/Statlog+%28Australian+Credit+Approval%29

def load_statlog_australian():
    path = 'Datasets/Raw/UCI_statlog_australian_credit_data_data_set/australian.dat'

    col_names = [f'A{i}' for i in range(1, 16)]

    cat_cols = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
    num_cols = [c for c in col_names if c not in cat_cols and c != 'A15']

    target_col = 'A15'

    df = pd.read_csv(path, sep=' ', header=None, index_col=False,
                     names=col_names,
                     dtype={col: 'category' for col in cat_cols})

    return df, cat_cols, num_cols, target_col


# #### Thomas2002
# L.C. Thomas, D.B. Edelman, J.N. Crook, Credit Scoring and its Applications, SIAM, Philadelphia, 2002.
# https://github.com/JLZml/Credit-Scoring-Data-Sets/blob/master/5.%20thomas/Loan%20Data.csv

def load_thomas():
    path = 'Datasets/Raw/Thomas_et_al_data_set/Loan Data.csv'

    cat_cols = ['PHON', 'AES', 'RES']

    target_col = 'BAD'

    df = pd.read_csv(path, sep=';', index_col=False,
                     dtype={col: 'category' for col in cat_cols})

    num_cols = [c for c in df.columns if c not in cat_cols and c != target_col]

    return df, cat_cols, num_cols, target_col


# ####  PAKDD2010
# https://github.com/JLZml/Credit-Scoring-Data-Sets/blob/master/2.%20PAKDD%202009%20Data%20Mining%20Competition/PAKDD%202010.zip
# http://sede.neurotech.com.br:443/PAKDD2009/

def load_pakdd():
    path = 'Datasets/Raw/PAKDD2010_data_set/PAKDD2010_Modeling_Data.txt'

    columns = ["ID_CLIENT", "CLERK_TYPE", "PAYMENT_DAY", "APPLICATION_SUBMISSION_TYPE", "QUANT_ADDITIONAL_CARDS",
               "POSTAL_ADDRESS_TYPE", "SEX", "MARITAL_STATUS", "QUANT_DEPENDANTS", "EDUCATION_LEVEL", "STATE_OF_BIRTH",
               "CITY_OF_BIRTH", "NACIONALITY", "RESIDENCIAL_STATE", "RESIDENCIAL_CITY", "RESIDENCIAL_BOROUGH",
               "FLAG_RESIDENCIAL_PHONE", "RESIDENCIAL_PHONE_AREA_CODE", "RESIDENCE_TYPE", "MONTHS_IN_RESIDENCE",
               "FLAG_MOBILE_PHONE", "FLAG_EMAIL", "PERSONAL_MONTHLY_INCOME", "OTHER_INCOMES", "FLAG_VISA",
               "FLAG_MASTERCARD", "FLAG_DINERS", "FLAG_AMERICAN_EXPRESS", "FLAG_OTHER_CARDS", "QUANT_BANKING_ACCOUNTS",
               "QUANT_SPECIAL_BANKING_ACCOUNTS", "PERSONAL_ASSETS_VALUE", "QUANT_CARS", "COMPANY", "PROFESSIONAL_STATE",
               "PROFESSIONAL_CITY", "PROFESSIONAL_BOROUGH", "FLAG_PROFESSIONAL_PHONE", "PROFESSIONAL_PHONE_AREA_CODE",
               "MONTHS_IN_THE_JOB", "PROFESSION_CODE", "OCCUPATION_TYPE", "MATE_PROFESSION_CODE",
               "MATE_EDUCATION_LEVEL", "FLAG_HOME_ADDRESS_DOCUMENT", "FLAG_RG", "FLAG_CPF", "FLAG_INCOME_PROOF",
               "PRODUCT", "FLAG_ACSP_RECORD", "AGE", "RESIDENCIAL_ZIP_3", "PROFESSIONAL_ZIP_3", "TARGET_BAD"]

    cat_cols = ['PAYMENT_DAY', 'APPLICATION_SUBMISSION_TYPE', 'POSTAL_ADDRESS_TYPE', 'SEX', 'MARITAL_STATUS',
                'STATE_OF_BIRTH', 'NACIONALITY', 'RESIDENCIAL_STATE', 'FLAG_RESIDENCIAL_PHONE',
                'RESIDENCIAL_PHONE_AREA_CODE', 'RESIDENCE_TYPE', 'FLAG_EMAIL', 'FLAG_VISA', 'FLAG_MASTERCARD',
                'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS', 'QUANT_BANKING_ACCOUNTS',
                'QUANT_SPECIAL_BANKING_ACCOUNTS', 'COMPANY', 'PROFESSIONAL_STATE', 'FLAG_PROFESSIONAL_PHONE',
                'PROFESSIONAL_PHONE_AREA_CODE', 'PROFESSION_CODE', 'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE',
                'MATE_EDUCATION_LEVEL', 'PRODUCT']

    num_cols = ['PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES', 'PERSONAL_ASSETS_VALUE', 'AGE', 'MONTHS_IN_RESIDENCE',
                'QUANT_DEPENDANTS', 'QUANT_CARS', 'MONTHS_IN_THE_JOB']

    target_col = 'TARGET_BAD'

    drop_cols = ['CITY_OF_BIRTH', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH', 'PROFESSIONAL_CITY',
                 'PROFESSIONAL_BOROUGH', 'RESIDENCIAL_ZIP_3', 'PROFESSIONAL_ZIP_3', 'FLAG_HOME_ADDRESS_DOCUMENT',
                 'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF', 'FLAG_ACSP_RECORD', 'CLERK_TYPE', 'QUANT_ADDITIONAL_CARDS',
                 'EDUCATION_LEVEL', 'FLAG_MOBILE_PHONE']

    df = pd.read_csv(path, sep='\t',
                     index_col='ID_CLIENT', encoding='unicode_escape',
                     header=None, names=columns,
                     dtype={col: 'category' for col in cat_cols}).drop(drop_cols, axis=1)

    return df, cat_cols, num_cols, target_col


# #### Taiwan
# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

def load_taiwan():
    path = 'Datasets/Raw/UCI_taiwan_default_of_credit_card_clients_data_set/default of credit card clients.csv'

    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    target_col = 'default payment next month'

    df = pd.read_csv(path, index_col=0,
                     dtype={col: 'category' for col in cat_cols})

    num_cols = [c for c in df.columns if c not in cat_cols and c != target_col]

    return df, cat_cols, num_cols, target_col


# #### bank
# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

def load_bank():
    path = 'Datasets/Raw/UCI_bank_marketing_data_set/bank-additional-full.csv'

    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                'contact', 'month', 'day_of_week', 'poutcome']
    num_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

    target_col = 'y'

    df = pd.read_csv(path, sep=';', index_col=False,
                     dtype={col: 'category' for col in cat_cols})

    df['y'] = np.where(df['y'] == 'yes', 1, 0)

    return df, cat_cols, num_cols, target_col


# #### homeeq
# http://www.creditriskanalytics.net/datasets-private2.html

def load_homeeq():
    path = 'Datasets/Raw/CREDITRISKANALYTICS_home_equity_data_set/hmeq.csv'

    cat_cols = ['REASON', 'JOB']
    num_cols = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG',
                'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']

    target_col = 'BAD'

    df = pd.read_csv(path, sep=',', index_col=False,
                     dtype={col: 'category' for col in cat_cols})

    return df, cat_cols, num_cols, target_col


# #### lending club 3a
# https://www.lendingclub.com/info/download-data.action

def load_lendingcluba():
    path = 'Datasets/Raw/Lending_Club_data_sets/LoanStats3a.csv'

    cat_cols = ['debt_settlement_flag', 'term', 'pub_rec_bankruptcies', 'verification_status', 'loan_status',
                'home_ownership', 'pub_rec', 'grade', 'emp_length', 'purpose', 'sub_grade', 'title']
    num_cols = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'annual_inc', 'revol_util', 'dti',
                'installment', 'inq_last_6mths', 'open_acc', 'revol_bal', 'total_acc', 'total_pymnt', 'total_pymnt_inv',
                'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                'last_pymnt_amnt']

    # many NANs and only one unique value present, or very few
    drop_cols = ['mths_since_last_delinq', 'mths_since_last_record', 'next_pymnt_d', 'settlement_term',
                 'settlement_amount', 'settlement_date', 'settlement_status', 'debt_settlement_flag_date',
                 'settlement_percentage', 'id', 'tot_coll_amt', 'sec_app_num_rev_accts', 'sec_app_open_act_il',
                 'sec_app_revol_util', 'sec_app_open_acc', 'sec_app_mort_acc', 'sec_app_inq_last_6mths',
                 'sec_app_earliest_cr_line', 'sec_app_chargeoff_within_12_mths', 'revol_bal_joint', 'total_bc_limit',
                 'total_bal_ex_mort', 'tot_hi_cred_lim', 'percent_bc_gt_75', 'pct_tl_nvr_dlq', 'num_tl_op_past_12m',
                 'num_tl_90g_dpd_24m', 'total_il_high_credit_limit', 'sec_app_collections_12_mths_ex_med',
                 'sec_app_mths_since_last_major_derog', 'hardship_type', 'member_id', 'url',
                 'mths_since_last_major_derog', 'annual_inc_joint', 'hardship_last_payment_amount',
                 'hardship_payoff_balance_amount', 'orig_projected_additional_accrued_interest', 'hardship_loan_status',
                 'hardship_dpd', 'hardship_length', 'payment_plan_start_date', 'hardship_end_date',
                 'hardship_start_date', 'hardship_amount', 'deferral_term', 'hardship_status', 'hardship_reason',
                 'num_tl_30dpd', 'verification_status_joint', 'num_tl_120dpd_2m', 'num_rev_tl_bal_gt_0', 'inq_last_12m',
                 'total_cu_tl', 'inq_fi', 'total_rev_hi_lim', 'all_util', 'max_bal_bc', 'open_rv_24m',
                 'acc_open_past_24mths', 'open_rv_12m', 'total_bal_il', 'mths_since_rcnt_il', 'open_il_24m',
                 'open_il_12m', 'open_act_il', 'open_acc_6m', 'tot_cur_bal', 'il_util', 'avg_cur_bal', 'bc_open_to_buy',
                 'bc_util', 'num_rev_accts', 'num_op_rev_tl', 'dti_joint', 'num_bc_tl', 'num_bc_sats',
                 'num_actv_rev_tl', 'num_actv_bc_tl', 'num_accts_ever_120_pd', 'mths_since_recent_revol_delinq',
                 'mths_since_recent_inq', 'mths_since_recent_bc_dlq', 'mths_since_recent_bc', 'mort_acc',
                 'mo_sin_rcnt_tl', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_old_rev_tl_op', 'mo_sin_old_il_acct', 'num_sats',
                 'num_il_tl',
                 'policy_code', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'collections_12_mths_ex_med',
                 'initial_list_status', 'application_type', 'hardship_flag', 'chargeoff_within_12_mths',
                 'acc_now_delinq', 'tax_liens', 'delinq_amnt',
                 'emp_title', 'zip_code', 'addr_state', 'last_pymnt_d', 'last_credit_pull_d', 'desc',
                 'earliest_cr_line', 'issue_d']

    target_col = 'delinq_2yrs'

    df = pd.read_csv(path, sep=',', index_col=False,
                     dtype={col: 'category' for col in cat_cols}
                     ).drop(drop_cols, axis=1)

    df['pub_rec_bankruptcies'] = (df['pub_rec_bankruptcies'].astype(float) > 0).astype('category')
    df['pub_rec'] = (df['pub_rec'].astype(float) > 0).astype('category')

    df['title'] = np.where(df['title'].isin(df.title.value_counts()[df.title.value_counts() > 100].index.values),
                           df['title'], 'OTHER')
    df['title'] = df['title'].astype('category')

    df['int_rate'] = df['int_rate'].apply(lambda x: float(x[:-1]) if not isinstance(x, np.float) else x)
    df['revol_util'] = df['revol_util'].apply(lambda x: float(x[:-1]) if not isinstance(x, np.float) else x)

    df = df.loc[~df[target_col].isna()]
    df[target_col] = (df[target_col] > 0).astype(int)

    return df, cat_cols, num_cols, target_col


# #### lending club 3b
# https://www.lendingclub.com/info/download-data.action

def load_lendingclubb():
    path = 'Datasets/Raw/Lending_Club_data_sets/LoanStats3b.csv'

    cat_cols = ['debt_settlement_flag', 'term', 'pub_rec_bankruptcies', 'verification_status', 'loan_status',
                'home_ownership', 'pub_rec', 'grade', 'emp_length', 'purpose', 'sub_grade', 'title']
    num_cols = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'annual_inc', 'revol_util', 'dti',
                'installment', 'inq_last_6mths', 'open_acc', 'revol_bal', 'total_acc', 'total_pymnt', 'total_pymnt_inv',
                'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                'last_pymnt_amnt']

    # many NANs and only one unique value present, or very few
    drop_cols = ['mths_since_last_delinq', 'mths_since_last_record', 'next_pymnt_d', 'settlement_term',
                 'settlement_amount', 'settlement_date', 'settlement_status', 'debt_settlement_flag_date',
                 'settlement_percentage', 'id', 'tot_coll_amt', 'sec_app_num_rev_accts', 'sec_app_open_act_il',
                 'sec_app_revol_util', 'sec_app_open_acc', 'sec_app_mort_acc', 'sec_app_inq_last_6mths',
                 'sec_app_earliest_cr_line', 'sec_app_chargeoff_within_12_mths', 'revol_bal_joint', 'total_bc_limit',
                 'total_bal_ex_mort', 'tot_hi_cred_lim', 'percent_bc_gt_75', 'pct_tl_nvr_dlq', 'num_tl_op_past_12m',
                 'num_tl_90g_dpd_24m', 'total_il_high_credit_limit', 'sec_app_collections_12_mths_ex_med',
                 'sec_app_mths_since_last_major_derog', 'hardship_type', 'member_id', 'url',
                 'mths_since_last_major_derog', 'annual_inc_joint', 'hardship_last_payment_amount',
                 'hardship_payoff_balance_amount', 'orig_projected_additional_accrued_interest', 'hardship_loan_status',
                 'hardship_dpd', 'hardship_length', 'payment_plan_start_date', 'hardship_end_date',
                 'hardship_start_date', 'hardship_amount', 'deferral_term', 'hardship_status', 'hardship_reason',
                 'num_tl_30dpd', 'verification_status_joint', 'num_tl_120dpd_2m', 'num_rev_tl_bal_gt_0', 'inq_last_12m',
                 'total_cu_tl', 'inq_fi', 'total_rev_hi_lim', 'all_util', 'max_bal_bc', 'open_rv_24m',
                 'acc_open_past_24mths', 'open_rv_12m', 'total_bal_il', 'mths_since_rcnt_il', 'open_il_24m',
                 'open_il_12m', 'open_act_il', 'open_acc_6m', 'tot_cur_bal', 'il_util', 'avg_cur_bal', 'bc_open_to_buy',
                 'bc_util', 'num_rev_accts', 'num_op_rev_tl', 'dti_joint', 'num_bc_tl', 'num_bc_sats',
                 'num_actv_rev_tl', 'num_actv_bc_tl', 'num_accts_ever_120_pd', 'mths_since_recent_revol_delinq',
                 'mths_since_recent_inq', 'mths_since_recent_bc_dlq', 'mths_since_recent_bc', 'mort_acc',
                 'mo_sin_rcnt_tl', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_old_rev_tl_op', 'mo_sin_old_il_acct', 'num_sats',
                 'num_il_tl',
                 'policy_code', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'collections_12_mths_ex_med',
                 'initial_list_status', 'application_type', 'hardship_flag', 'chargeoff_within_12_mths',
                 'acc_now_delinq', 'tax_liens', 'delinq_amnt',
                 'emp_title', 'zip_code', 'addr_state', 'last_pymnt_d', 'last_credit_pull_d', 'desc',
                 'earliest_cr_line', 'issue_d']

    target_col = 'delinq_2yrs'

    df = pd.read_csv(path, sep=',', index_col=False,
                     dtype={col: 'category' for col in cat_cols}
                     ).drop(drop_cols, axis=1)

    df['pub_rec_bankruptcies'] = (df['pub_rec_bankruptcies'].astype(float) > 0).astype('category')
    df['pub_rec'] = (df['pub_rec'].astype(float) > 0).astype('category')

    df['title'] = np.where(df['title'].isin(df.title.value_counts()[df.title.value_counts() > 100].index.values),
                           df['title'], 'OTHER')
    df['title'] = df['title'].astype('category')

    df['int_rate'] = df['int_rate'].apply(lambda x: float(x[:-1]) if not isinstance(x, np.float) else x)
    df['revol_util'] = df['revol_util'].apply(lambda x: float(x[:-1]) if not isinstance(x, np.float) else x)

    df = df.loc[~df[target_col].isna()]
    df[target_col] = (df[target_col] > 0).astype(int)

    return df, cat_cols, num_cols, target_col


# #### Give me credit
# https://www.kaggle.com/c/GiveMeSomeCredit/data?select=cs-training.csv

def load_gmc():
    path = 'Datasets/Raw/Kaggle_give_me_credit_data_set/cs-training.csv'

    cat_cols = []

    target_col = 'SeriousDlqin2yrs'

    df = pd.read_csv(path, sep=',', index_col=0,
                     dtype={col: 'category' for col in cat_cols})

    num_cols = [c for c in df.columns if c != target_col]

    return df, cat_cols, num_cols, target_col


# #### dmc05
# https://www.data-mining-cup.com/reviews/dmc-2005/


def load_dmc05():
    path = 'Datasets/Raw/DMC05_ecommerce_fraud_data_set/dmc2005_train.txt'

    cat_cols = ['B_EMAIL', 'B_TELEFON', 'B_GEBDATUM', 'FLAG_LRIDENTISCH', 'FLAG_NEWSLETTER',
                'Z_METHODE', 'Z_CARD_ART', 'Z_LAST_NAME', 'TAG_BEST', 'TIME_BEST', 'CHK_LADR',
                'CHK_RADR', 'CHK_KTO', 'CHK_CARD', 'CHK_COOKIE', 'CHK_IP', 'FAIL_LPLZ',
                'FAIL_LORT', 'FAIL_LPLZORTMATCH', 'FAIL_RPLZ', 'FAIL_RORT',
                'FAIL_RPLZORTMATCH', 'NEUKUNDE', 'DATUM_LBEST']

    num_cols = ['Z_CARD_VALID', 'WERT_BEST', 'ANZ_BEST', 'ANUMMER_01', 'ANUMMER_02',
                'ANUMMER_03', 'ANUMMER_04', 'ANUMMER_05', 'ANUMMER_06', 'ANUMMER_07',
                'ANUMMER_08', 'ANUMMER_09', 'ANUMMER_10', 'SESSION_TIME', 'ANZ_BEST_GES',
                'WERT_BEST_GES', 'MAHN_AKT', 'MAHN_HOECHST']

    target_col = 'TARGET_BETRUG'

    df = pd.read_csv(path, sep='\t', index_col='BESTELLIDENT',
                     dtype={col: 'category' for col in cat_cols})

    df['TARGET_BETRUG'] = np.where(df['TARGET_BETRUG'] == 'ja', 1, 0)

    return df, cat_cols, num_cols, target_col


# #### dmc10
# https://www.data-mining-cup.com/reviews/dmc-2010/


def load_dmc10():
    path = 'Datasets/Raw/DMC10_ecommerce_voucher_data_set/dmc2010_train.txt'

    cat_cols = ['delivpostcode', 'advertisingdatacode', 'salutation', 'title',
                'domain', 'newsletter', 'model', 'paymenttype', 'deliverytype',
                'invoicepostcode', 'voucher', 'case', 'gift', 'entry', 'points',
                'shippingcosts']
    num_cols = ['numberitems', 'weight', 'remi', 'cancel', 'used', 'w0', 'w1',
                'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'account_age']
    drop_cols = ['date', 'datecreated', 'deliverydatepromised', 'deliverydatereal']

    target_col = 'target90'

    df = pd.read_csv(path, sep=';', index_col='customernumber', parse_dates=drop_cols,
                     dtype={col: 'category' for col in cat_cols})
    df['account_age'] = (df['date'] - df['datecreated']).dt.days

    df.drop(drop_cols, axis=1, inplace=True)

    return df, cat_cols, num_cols, target_col


# #### coil2k
# http://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)


def load_coil2k():
    path = 'Datasets/Raw/UCI_coil2k_insurance_data_set/ticdata2000.txt'

    col_names = ['MOSTYPE', 'MAANTHUI', 'MGEMOMV', 'MGEMLEEF', 'MOSHOOFD', 'MGODRK',
                 'MGODPR', 'MGODOV', 'MGODGE', 'MRELGE', 'MRELSA', 'MRELOV', 'MFALLEEN',
                 'MFGEKIND', 'MFWEKIND', 'MOPLHOOG', 'MOPLMIDD', 'MOPLLAAG', 'MBERHOOG',
                 'MBERZELF', 'MBERBOER', 'MBERMIDD', 'MBERARBG', 'MBERARBO', 'MSKA',
                 'MSKB1', 'MSKB2', 'MSKC', 'MSKD', 'MHHUUR', 'MHKOOP', 'MAUT1', 'MAUT2',
                 'MAUT0', 'MZFONDS', 'MZPART', 'MINKM30', 'MINK3045', 'MINK4575',
                 'MINK7512', 'MINK123M', 'MINKGEM', 'MKOOPKLA', 'PWAPART', 'PWABEDR',
                 'PWALAND', 'PPERSAUT', 'PBESAUT', 'PMOTSCO', 'PVRAAUT', 'PAANHANG',
                 'PTRACTOR', 'PWERKT', 'PBROM', 'PLEVEN', 'PPERSONG', 'PGEZONG', 'PWAOREG',
                 'PBRAND', 'PZEILPL', 'PPLEZIER', 'PFIETS', 'PINBOED', 'PBYSTAND',
                 'AWAPART', 'AWABEDR', 'AWALAND', 'APERSAUT', 'ABESAUT', 'AMOTSCO',
                 'AVRAAUT', 'AAANHANG', 'ATRACTOR', 'AWERKT', 'ABROM', 'ALEVEN', 'APERSONG',
                 'AGEZONG', 'AWAOREG', 'ABRAND', 'AZEILPL', 'APLEZIER', 'AFIETS',
                 'AINBOED', 'ABYSTAND', 'CARAVAN']

    cat_cols = ['MOSTYPE', 'MGEMLEEF', 'MOSHOOFD', 'MKOOPKLA', 'PWAPART',
                'PWABEDR', 'PWALAND', 'PPERSAUT', 'PBESAUT', 'PMOTSCO',
                'PVRAAUT', 'PAANHANG', 'PTRACTOR', 'PWERKT', 'PBROM',
                'PLEVEN', 'PPERSONG', 'PGEZONG', 'PWAOREG', 'PBRAND',
                'PZEILPL', 'PPLEZIER', 'PFIETS', 'PINBOED', 'PBYSTAND', 'AWAPART']
    num_cols = ['MAANTHUI', 'MGEMOMV', 'MGODRK', 'MGODPR', 'MGODOV', 'MGODGE', 'MRELGE',
                'MRELSA', 'MRELOV', 'MFALLEEN', 'MFGEKIND', 'MFWEKIND', 'MOPLHOOG', 'MOPLMIDD',
                'MOPLLAAG', 'MBERHOOG', 'MBERZELF', 'MBERBOER', 'MBERMIDD', 'MBERARBG', 'MBERARBO',
                'MSKA', 'MSKB1', 'MSKB2', 'MSKC', 'MSKD', 'MHHUUR', 'MHKOOP', 'MAUT1', 'MAUT2',
                'MAUT0', 'MZFONDS', 'MZPART', 'MINKM30', 'MINK3045', 'MINK4575', 'MINK7512', 'MINK123M',
                'MINKGEM', 'AWABEDR', 'AWALAND', 'APERSAUT', 'ABESAUT', 'AMOTSCO', 'AVRAAUT', 'AAANHANG',
                'ATRACTOR', 'AWERKT', 'ABROM', 'ALEVEN', 'APERSONG', 'AGEZONG', 'AWAOREG', 'ABRAND',
                'AZEILPL', 'APLEZIER', 'AFIETS', 'AINBOED', 'ABYSTAND']
    target_col = 'CARAVAN'

    df = pd.read_csv(path, sep='\t', header=None,
                     names=col_names,
                     dtype={col: 'category' for col in cat_cols})

    extra_path_X = 'Datasets/Raw/UCI_coil2k_insurance_data_set/ticeval2000.txt'
    extra_df = pd.read_csv(extra_path_X, sep='\t', header=None,
                           names=col_names[:-1],
                           dtype={col: 'category' for col in cat_cols})
    extra_path_y = 'Datasets/Raw/UCI_coil2k_insurance_data_set/tictgts2000.txt'
    extra_df[target_col] = pd.read_csv(extra_path_y, sep='\t', header=None)

    df = df.append(extra_df).reset_index(drop=True)

    df[cat_cols] = df[cat_cols].astype('category')

    return df, cat_cols, num_cols, target_col


# #### adult
# https://archive.ics.uci.edu/ml/datasets/adult


def load_adult():
    path = 'Datasets/Raw/UCI_adult_data_set/adult.data'

    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                 'marital-status', 'occupation', 'relationship',
                 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                 'native-country', 'target']

    cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']
    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    target_col = 'target'

    df = pd.read_csv(path, sep=',', index_col=None, names=col_names,
                     dtype={col: 'category' for col in cat_cols})

    df[target_col] = np.where(df[target_col] == ' >50K', 1, 0)

    return df, cat_cols, num_cols, target_col


# #### forest
# https://archive.ics.uci.edu/ml/datasets/covertype
# Work in progress / unfinished
def load_forest():
    path = 'Datasets/Raw/UCI_forest_covertype_data_set/covtype.data'

    #     col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
    #                  'marital-status', 'occupation', 'relationship',
    #                  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
    #                  'native-country', 'target']

    #     cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
    #                 'relationship', 'race', 'sex', 'native-country']
    #     num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    #     target_col = 'target'

    df = pd.read_csv(path, sep=',', index_col=None)  # , names=col_names,
    #                      dtype={col: 'category' for col in cat_cols})

    #     df[target_col] = np.where(df[target_col]== ' >50K', 1, 0)

    return df  # , cat_cols, num_cols, target_col
