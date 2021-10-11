# support.py

data_dir = 'data/'
model_dir = 'models/'
eval_dir = 'eval_data/'


selected_cols = [
    'senior_citizen',
    'partner',
    'dependents',
    'online_security',
    'tech_support',
    'tenure',
    'paperless_billing',
    'monthly_charges',
    'internet_service_fiber_optic',
    'internet_service_no',
    'contract_month_to_month',
    'contract_one_year',
    'contract_two_year',
    'payment_method_electronic_check',
    'churn'
]


logreg_params = {
    'm__C': [0.01, 0.1, 1, 10, 100],
    'm__tol': [0.0001, 0.001, 0.001, 0.01],
    'm__class_weight': ['balanced', None],
    'm__max_iter': [100, 200, 500]
}
