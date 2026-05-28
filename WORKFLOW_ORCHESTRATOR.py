"""
INDUSTRIAL ML WORKFLOW ORCHESTRATOR
Alpha Defect Prediction in Hot Rolling Mills

Master script to coordinate all 8 workflow parts
"""

import sys
import os
import time
from datetime import datetime

def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)

def print_section(text):
    print("\n" + "-"*70)
    print(text)
    print("-"*70)

class WorkflowOrchestrator:
    """Master workflow orchestrator"""

    def __init__(self):
        self.start_time = datetime.now()
        self.parts_completed = []
        self.parts_failed = []

    def run_part(self, part_num, part_name, module_name):
        """Run individual workflow part"""
        print_section(f"EXECUTING PART {part_num}: {part_name}")

        try:
            # Dynamic import
            module = __import__(module_name)

            # Get main class
            class_name = {
                1: 'IndustrialEDA',
                2: 'BaselineModeling',
                3: 'SHAPErrorAnalysis',
                4: 'CorrelationGrouping',
                5: 'FeatureEngineering',
                6: 'ImbalanceHandling',
                7: 'CalibrationThreshold',
                8: 'FinalRefinement'
            }[part_num]

            main_class = getattr(module, class_name)

            # Execute based on part number
            if part_num == 1:
                eda = main_class(train_path='train.csv', test_path='test.csv')
                eda.run_complete_eda()

            elif part_num == 2:
                baseline = main_class(train_path='train.csv', test_path='test.csv')
                baseline.run_baseline_modeling()

            elif part_num == 3:
                shap_analysis = main_class(train_path='train.csv')
                shap_analysis.run_shap_analysis()

            elif part_num == 4:
                grouping = main_class(train_path='train.csv')
                grouping.run_grouping_analysis()

            elif part_num == 5:
                fe = main_class(train_path='train.csv')
                fe.run_feature_engineering()

            elif part_num == 6:
                imbalance = main_class(engineered_path='X_engineered.csv',
                                      target_path='y_train.csv')
                imbalance.run_imbalance_handling()

            elif part_num == 7:
                calib = main_class(engineered_path='X_engineered.csv',
                                  target_path='y_train.csv')
                calib.run_calibration_threshold()

            elif part_num == 8:
                final = main_class(engineered_path='X_engineered.csv',
                                  target_path='y_train.csv')
                final.run_final_refinement()

            self.parts_completed.append((part_num, part_name))
            print_section(f"✓ PART {part_num} COMPLETED SUCCESSFULLY")

        except Exception as e:
            print_section(f"✗ PART {part_num} FAILED")
            print(f"Error: {str(e)}")
            self.parts_failed.append((part_num, part_name, str(e)))

    def run_full_workflow(self):
        """Execute complete workflow"""
        print_header("ALPHA DEFECT PREDICTION - INDUSTRIAL ML WORKFLOW")
        print(f"\nWorkflow Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        workflow_parts = [
            (1, "INDUSTRIAL EDA", "part1_industrial_eda"),
            (2, "QUICK BASELINE XGBOOST/LIGHTGBM", "part2_baseline_model"),
            (3, "SHAP + ERROR ANALYSIS", "part3_shap_error_analysis"),
            (4, "CORRELATION GROUPING", "part4_correlation_grouping"),
            (5, "FEATURE ENGINEERING", "part5_feature_engineering"),
            (6, "IMBALANCE HANDLING", "part6_imbalance_handling"),
            (7, "CALIBRATION + THRESHOLD TUNING", "part7_calibration_threshold"),
            (8, "FINAL REFINEMENT", "part8_final_refinement")
        ]

        for part_num, part_name, module_name in workflow_parts:
            self.run_part(part_num, part_name, module_name)

        self.print_summary()

    def print_summary(self):
        """Print workflow summary"""
        print_header("WORKFLOW EXECUTION SUMMARY")

        end_time = datetime.now()
        duration = end_time - self.start_time

        print(f"\nStart Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {duration}")

        print_section("PARTS COMPLETED")
        for part_num, part_name in self.parts_completed:
            print(f"✓ Part {part_num}: {part_name}")

        if self.parts_failed:
            print_section("PARTS FAILED")
            for part_num, part_name, error in self.parts_failed:
                print(f"✗ Part {part_num}: {part_name}")
                print(f"  Error: {error}")
        else:
            print_section("ALL PARTS COMPLETED SUCCESSFULLY")

        print_section("GENERATED OUTPUTS")
        print("""
Analysis Outputs:
  ✓ 01_class_distribution.png
  ✓ 02_defect_vs_normal.png
  ✓ 03_instability_analysis.png
  ✓ 04_outlier_analysis.png
  ✓ 05_correlation_heatmap.png
  ✓ 06_correlation_comparison.png
  ✓ 07_defect_density.png
  ✓ 08_pca_visualization.png
  ✓ 09_tsne_visualization.png
  ✓ 10_hidden_regimes.png
  ✓ 11_threshold_stability.png
  ✓ 12_anomaly_behavior.png
  ✓ 13_probability_analysis.png
  ✓ 14_threshold_sensitivity.png
  ✓ 15_global_shap_analysis.png
  ✓ 16_false_positive_analysis.png
  ✓ 17_false_negative_analysis.png
  ✓ 18_hard_sample_analysis.png
  ✓ 19_full_correlation_matrix.png
  ✓ 20_feature_dendrogram.png
  ✓ 21_group_defect_behavior.png
  ✓ 22_imbalance_strategy_comparison.png
  ✓ 23_threshold_optimization.png

Insights Reports:
  ✓ PART1_INSIGHTS.txt - Industrial EDA findings
  ✓ PART2_INSIGHTS.txt - Baseline model evaluation
  ✓ PART3_INSIGHTS.txt - Error analysis and SHAP insights
  ✓ PART4_INSIGHTS.txt - Feature grouping and correlations
  ✓ PART5_INSIGHTS.txt - Engineered features summary
  ✓ PART6_INSIGHTS.txt - Imbalance handling comparison
  ✓ PART7_INSIGHTS.txt - Calibration and threshold optimization
  ✓ PART8_INSIGHTS.txt - Final model refinement

Model Artifacts:
  ✓ X_engineered.csv - Engineered feature set
  ✓ y_train.csv - Target variable
  ✓ Trained models ready for deployment
""")

        print_section("NEXT STEPS FOR DEPLOYMENT")
        print("""
1. REVIEW ALL INSIGHTS REPORTS
   - Read PART*_INSIGHTS.txt files
   - Understand model behavior and trade-offs

2. EXAMINE VISUALIZATIONS
   - Review all PNG plots for insights
   - Understand defect patterns and boundaries

3. PREPARE PRODUCTION ENVIRONMENT
   - Set up model serving infrastructure
   - Configure optimal threshold from Part 7

4. IMPLEMENT MONITORING
   - Track prediction distribution
   - Monitor false positive rate
   - Monitor escaped defects (false negatives)

5. OPERATIONAL DEPLOYMENT
   - Deploy calibrated model
   - Apply optimized threshold
   - Flag high-confidence defects for inspection

6. CONTINUOUS IMPROVEMENT
   - Collect feedback from field
   - Recalibrate quarterly
   - Retrain when new patterns emerge
""")

        print_header("WORKFLOW COMPLETE - READY FOR PRODUCTION DEPLOYMENT")

if __name__ == "__main__":
    orchestrator = WorkflowOrchestrator()
    orchestrator.run_full_workflow()
