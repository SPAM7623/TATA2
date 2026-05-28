"""
INSIGHTS MANAGER - Central repository for cross-part knowledge propagation
Enables vertical knowledge flow: insights from Part N inform decisions in Part N+1, N+2, etc.
"""

import json
import pickle
import os
from pathlib import Path

class InsightsManager:
    """Manages insights across all workflow parts"""

    def __init__(self, filepath='workflow_insights.pkl'):
        self.filepath = filepath
        self.insights = {}
        self.load()

    def load(self):
        """Load existing insights from file"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'rb') as f:
                    self.insights = pickle.load(f)
                print(f"✓ Loaded insights from {self.filepath}")
            except Exception as e:
                print(f"⚠ Could not load insights: {e}")
                self.insights = {}
        else:
            self.insights = {}

    def save(self):
        """Save insights to file"""
        try:
            with open(self.filepath, 'wb') as f:
                pickle.dump(self.insights, f)
            print(f"✓ Saved insights to {self.filepath}")
        except Exception as e:
            print(f"✗ Error saving insights: {e}")

    # PART 1 INSIGHTS
    def set_part1_insights(self, insights_dict):
        """Part 1 saves its insights for downstream parts"""
        if 'part1' not in self.insights:
            self.insights['part1'] = {}

        self.insights['part1'].update({
            'class_imbalance_ratio': insights_dict.get('class_imbalance_ratio'),
            'unstable_features': insights_dict.get('unstable_features', [])[:10],  # Top 10
            'low_variance_features': insights_dict.get('low_variance_features', []),
            'high_corr_pairs': insights_dict.get('high_corr_pairs', [])[:10],
            'outlier_rate_defect': insights_dict.get('outlier_rate_defect'),
            'feature_groups': insights_dict.get('feature_groups', {}),
        })
        self.save()
        print("✓ Part 1 insights saved")

    def get_part1_insights(self):
        """Downstream parts retrieve Part 1 insights"""
        return self.insights.get('part1', {})

    # PART 2 INSIGHTS
    def set_part2_insights(self, insights_dict):
        """Part 2 saves baseline metrics"""
        if 'part2' not in self.insights:
            self.insights['part2'] = {}

        self.insights['part2'].update({
            'baseline_roc_auc': insights_dict.get('xgb_cv_results', {}).mean().get('roc_auc'),
            'baseline_pr_auc': insights_dict.get('xgb_cv_results', {}).mean().get('pr_auc'),
            'baseline_f1': insights_dict.get('xgb_cv_results', {}).mean().get('f1'),
            'xgb_importances': insights_dict.get('xgb_importances'),
        })
        self.save()
        print("✓ Part 2 insights saved")

    def get_part2_insights(self):
        """Retrieve Part 2 insights"""
        return self.insights.get('part2', {})

    # PART 3 INSIGHTS
    def set_part3_insights(self, insights_dict):
        """Part 3 saves error patterns"""
        if 'part3' not in self.insights:
            self.insights['part3'] = {}

        self.insights['part3'].update({
            'fn_rate': insights_dict.get('fn_rate'),  # False negative rate
            'fn_characteristics': insights_dict.get('fn_characteristics', []),
            'fp_rate': insights_dict.get('fp_rate'),
            'fp_characteristics': insights_dict.get('fp_characteristics', []),
            'shap_interactions': insights_dict.get('shap_interactions', [])[:5],
        })
        self.save()
        print("✓ Part 3 insights saved")

    def get_part3_insights(self):
        """Retrieve Part 3 error analysis insights"""
        return self.insights.get('part3', {})

    # PART 4 INSIGHTS
    def set_part4_insights(self, insights_dict):
        """Part 4 saves feature grouping"""
        if 'part4' not in self.insights:
            self.insights['part4'] = {}

        self.insights['part4'].update({
            'feature_groups': insights_dict.get('feature_groups', {}),
            'group_defect_analysis': insights_dict.get('group_defect_analysis', {}),
        })
        self.save()
        print("✓ Part 4 insights saved")

    def get_part4_insights(self):
        """Retrieve Part 4 feature groups"""
        return self.insights.get('part4', {})

    # PART 5 INSIGHTS
    def set_part5_insights(self, insights_dict):
        """Part 5 saves engineered feature stats"""
        if 'part5' not in self.insights:
            self.insights['part5'] = {}

        self.insights['part5'].update({
            'total_engineered_features': insights_dict.get('total_engineered_features'),
            'engineered_feature_importance': insights_dict.get('engineered_feature_importance'),
        })
        self.save()
        print("✓ Part 5 insights saved")

    def get_part5_insights(self):
        """Retrieve Part 5 engineered feature info"""
        return self.insights.get('part5', {})

    # PART 6 INSIGHTS
    def set_part6_insights(self, insights_dict):
        """Part 6 saves best imbalance strategy"""
        if 'part6' not in self.insights:
            self.insights['part6'] = {}

        strategy_comparison = insights_dict.get('strategy_comparison', {})
        best_f1_strategy = strategy_comparison.idxmax()['f1'] if hasattr(strategy_comparison, 'idxmax') else None
        best_recall_strategy = strategy_comparison.idxmax()['recall'] if hasattr(strategy_comparison, 'idxmax') else None

        self.insights['part6'].update({
            'best_f1_strategy': best_f1_strategy,
            'best_recall_strategy': best_recall_strategy,
            'strategy_metrics': strategy_comparison.to_dict() if hasattr(strategy_comparison, 'to_dict') else {},
        })
        self.save()
        print("✓ Part 6 insights saved")

    def get_part6_insights(self):
        """Retrieve Part 6 imbalance strategy selection"""
        return self.insights.get('part6', {})

    # PART 7 INSIGHTS
    def set_part7_insights(self, insights_dict):
        """Part 7 saves optimal threshold"""
        if 'part7' not in self.insights:
            self.insights['part7'] = {}

        optimal_thresholds = insights_dict.get('optimal_thresholds', {})

        self.insights['part7'].update({
            'optimal_threshold_f1': optimal_thresholds.get('best_f1'),
            'optimal_threshold_recall95': optimal_thresholds.get('recall_95'),
            'optimal_threshold_pr_balance': optimal_thresholds.get('pr_balance'),
            'threshold_sweep': insights_dict.get('threshold_sweep'),
        })
        self.save()
        print("✓ Part 7 insights saved - OPTIMAL THRESHOLD DETERMINED")

    def get_part7_insights(self):
        """Retrieve Part 7 threshold optimization"""
        return self.insights.get('part7', {})

    # PART 8 INSIGHTS
    def set_part8_insights(self, insights_dict):
        """Part 8 saves final model config"""
        if 'part8' not in self.insights:
            self.insights['part8'] = {}

        self.insights['part8'].update({
            'final_model_config': insights_dict.get('final_model_config', {}),
            'best_params': insights_dict.get('best_params', {}),
            'consistency_analysis': insights_dict.get('consistency_analysis'),
            'calibration_stability': insights_dict.get('calibration_stability'),
        })
        self.save()
        print("✓ Part 8 insights saved - FINAL MODEL READY")

    def get_part8_insights(self):
        """Retrieve Part 8 final model config"""
        return self.insights.get('part8', {})

    # SUMMARY METHODS
    def print_workflow_summary(self):
        """Print summary of all insights"""
        print("\n" + "="*70)
        print("WORKFLOW INSIGHTS SUMMARY")
        print("="*70)

        if 'part1' in self.insights:
            p1 = self.insights['part1']
            print(f"\n✓ PART 1 - Data Characteristics:")
            print(f"  Class imbalance ratio: {p1.get('class_imbalance_ratio'):.2f}:1")
            print(f"  Top unstable features: {p1.get('unstable_features', [])[:3]}")

        if 'part2' in self.insights:
            p2 = self.insights['part2']
            print(f"\n✓ PART 2 - Baseline Metrics:")
            print(f"  ROC-AUC: {p2.get('baseline_roc_auc'):.4f}")
            print(f"  F1: {p2.get('baseline_f1'):.4f}")

        if 'part3' in self.insights:
            p3 = self.insights['part3']
            print(f"\n✓ PART 3 - Error Patterns:")
            print(f"  False negative rate: {p3.get('fn_rate'):.2f}% (escaped defects)")
            print(f"  False positive rate: {p3.get('fp_rate'):.2f}%")

        if 'part4' in self.insights:
            p4 = self.insights['part4']
            print(f"\n✓ PART 4 - Feature Groups:")
            print(f"  Number of groups: {len(p4.get('feature_groups', {}))}")

        if 'part5' in self.insights:
            p5 = self.insights['part5']
            print(f"\n✓ PART 5 - Feature Engineering:")
            print(f"  Total engineered features: {p5.get('total_engineered_features')}")

        if 'part6' in self.insights:
            p6 = self.insights['part6']
            print(f"\n✓ PART 6 - Imbalance Strategy:")
            print(f"  Best strategy: {p6.get('best_f1_strategy')}")

        if 'part7' in self.insights:
            p7 = self.insights['part7']
            print(f"\n✓ PART 7 - Optimal Threshold:")
            print(f"  Threshold (F1 optimal): {p7.get('optimal_threshold_f1'):.4f}")
            print(f"  Threshold (95% recall): {p7.get('optimal_threshold_recall95'):.4f}")

        if 'part8' in self.insights:
            p8 = self.insights['part8']
            config = p8.get('final_model_config', {})
            print(f"\n✓ PART 8 - Final Model:")
            print(f"  Max depth: {config.get('max_depth')}")
            print(f"  Learning rate: {config.get('learning_rate')}")
            print(f"  N estimators: {config.get('n_estimators')}")

        print("\n" + "="*70)

    def export_to_json(self, filename='workflow_insights.json'):
        """Export insights to JSON for deployment"""
        export_dict = {}

        # Convert non-serializable objects
        for part_key, part_data in self.insights.items():
            export_dict[part_key] = {}
            for key, value in part_data.items():
                if hasattr(value, 'to_dict'):
                    export_dict[part_key][key] = value.to_dict()
                elif hasattr(value, 'to_list'):
                    export_dict[part_key][key] = value.to_list()
                elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    export_dict[part_key][key] = value
                else:
                    export_dict[part_key][key] = str(value)

        try:
            with open(filename, 'w') as f:
                json.dump(export_dict, f, indent=2)
            print(f"✓ Exported insights to {filename}")
        except Exception as e:
            print(f"✗ Error exporting: {e}")

if __name__ == "__main__":
    # Test the manager
    manager = InsightsManager()
    manager.print_workflow_summary()
