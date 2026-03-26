import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path='configs/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def simulate_current_data(reference_df, drift_intensity=0.2):
    current_df = reference_df.copy()
    n = len(current_df)
    numeric_cols = [c for c in current_df.columns if c.startswith('pixel_')]
    n_drifted = max(1, int(len(numeric_cols) * drift_intensity))
    drifted_cols = np.random.choice(numeric_cols, n_drifted, replace=False)
    for col in drifted_cols:
        noise = current_df[col].std() * 0.5
        current_df[col] = (current_df[col] + np.random.normal(0.3, noise, n)).clip(0, 1)
    logger.info(f'Derive simulee sur {n_drifted} colonnes')
    return current_df

def run_monitoring():
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.test_suite import TestSuite
        from evidently.test_preset import DataDriftTestPreset
    except ImportError:
        logger.error('Evidently non installe : pip install evidently==0.4.22')
        return
    config = load_config()
    ref_path = config['monitoring']['reference_data_path']
    out_path = config['monitoring']['report_output_path']
    if not os.path.exists(ref_path):
        logger.error(f'Donnees manquantes : {ref_path}')
        return
    logger.info('MONITORING EVIDENTLY AI')
    reference = pd.read_csv(ref_path)
    cols = [f'pixel_{i}' for i in range(0, 784, 40)] + ['target']
    reference = reference[[c for c in cols if c in reference.columns]]
    logger.info(f'Reference : {reference.shape}')
    current = simulate_current_data(reference, drift_intensity=0.2)
    current.to_csv(config['monitoring']['current_data_path'], index=False)
    os.makedirs(out_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(
        reference_data=reference.drop(columns=['target'], errors='ignore'),
        current_data=current.drop(columns=['target'], errors='ignore')
    )
    drift_path = os.path.join(out_path, f'drift_report_{timestamp}.html')
    report.save_html(drift_path)
    result = report.as_dict()
    drift_detected = result['metrics'][0]['result'].get('dataset_drift', False)
    drift_share = result['metrics'][0]['result'].get('share_of_drifted_columns', 0)
    test_suite = TestSuite(tests=[DataDriftTestPreset()])
    test_suite.run(
        reference_data=reference.drop(columns=['target'], errors='ignore'),
        current_data=current.drop(columns=['target'], errors='ignore')
    )
    test_path = os.path.join(out_path, f'test_suite_{timestamp}.html')
    test_suite.save_html(test_path)
    logger.info(f'Drift detecte : {"OUI" if drift_detected else "NON"}')
    logger.info(f'Colonnes driftees : {drift_share:.1%}')
    logger.info(f'Rapport : {drift_path}')
    if drift_detected:
        logger.warning('ALERTE : Derive detectee !')
    return drift_path

if __name__ == '__main__':
    run_monitoring()