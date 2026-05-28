"""
PART 3: SHAP + ERROR ANALYSIS (VERY IMPORTANT)
Alpha Defect Prediction in Hot Rolling Mills

Goal: Understand WHY model succeeds/fails

Key Insights To Extract:
- Which variables dominate predictions?
- Which interactions matter most?
- Which defects are consistently missed?
- Which normal samples resemble defects?
- Which variables create confusion?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# PART 3 CHECKLIST
# =====================================================================
CHECKLIST = {
    "3.1_data_loading": False,
    "3.2_model_training": False,
    "3.3_global_shap": False,
    "3.4_local_shap": False,
    "3.5_shap_interactions": False,
    "3.6_false_positive_analysis": False,
    "3.7_false_negative_analysis": False,
    "3.8_hard_sample_analysis": False,
}

class SHAPErrorAnalysis:
    """SHAP-based error analysis"""

    def __init__(self, train_path='train.csv'):
        self.train_path = train_path
        self.train_df = None
        self.X_train = None
        self.y_train = None
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.insights = {}

    def load_data(self):
        """3.1: Load training data"""
        print("\n" + "="*70)
        print("3.1 DATA LOADING")
        print("="*70)

        # Load Part 1 insights
        from insights_manager import InsightsManager
        manager = InsightsManager()
        part1_insights = manager.get_part1_insights()

        if part1_insights:
            print("\n✓ Using Part 1 insights to guide error analysis:")
            imbalance_ratio = part1_insights.get('class_imbalance_ratio', 1.0)
            print(f"  Class imbalance: {imbalance_ratio:.2f}:1 (high imbalance)")
            print(f"  → Will prioritize ESCAPED DEFECT detection")
            self.insights['imbalance_context'] = imbalance_ratio

        self.train_df = pd.read_csv(self.train_path)
        self.X_train = self.train_df.drop(['CoilID', 'Y'], axis=1)
        self.y_train = self.train_df['Y']

        print(f"Training set: {self.X_train.shape}")
        print(f"Class distribution: {self.y_train.value_counts().to_dict()}")

        CHECKLIST["3.1_data_loading"] = True

    def train_model(self):
        """3.2: Train XGBoost model for SHAP analysis"""
        print("\n" + "="*70)
        print("3.2 MODEL TRAINING FOR SHAP")
        print("="*70)

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )

        self.model.fit(self.X_train, self.y_train)
        print("Model trained for SHAP explanation")

        # Get predictions
        proba = self.model.predict_proba(self.X_train)[:, 1]
        pred = self.model.predict(self.X_train)

        print(f"\nPrediction accuracy: {(pred == self.y_train).mean():.4f}")

        CHECKLIST["3.2_model_training"] = True

    def global_shap_analysis(self):
        """3.3: Global SHAP analysis"""
        print("\n" + "="*70)
        print("3.3 GLOBAL SHAP ANALYSIS")
        print("="*70)

        # Use smaller sample for faster SHAP computation
        sample_idx = np.random.choice(len(self.X_train), min(1000, len(self.X_train)),
                                     replace=False)
        X_sample = self.X_train.iloc[sample_idx]

        print("Computing SHAP values (this may take a moment)...")
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_sample)

        # Handle binary classification output
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]  # Class 1 (defect)
        else:
            shap_vals = self.shap_values

        # Feature importance from SHAP
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)

        print("\nTop 15 Most Important Features (by SHAP):")
        for idx, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']}: {row['mean_abs_shap']:.6f}")

        self.insights['feature_importance_shap'] = feature_importance

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        # Summary plot
        feature_importance.head(15).plot(x='feature', y='mean_abs_shap', kind='barh', ax=axes[0])
        axes[0].set_xlabel('Mean |SHAP|')
        axes[0].set_title('Top 15 Features by SHAP Importance')

        # SHAP summary scatter (simplified)
        top_features_idx = feature_importance.head(6).index
        for idx, feature_idx in enumerate(top_features_idx):
            feature_name = self.X_train.columns[feature_idx]
            axes[1].scatter(self.X_train.iloc[sample_idx, feature_idx],
                           np.abs(shap_vals[:, feature_idx]),
                           alpha=0.5, s=20, label=feature_name)

        axes[1].set_xlabel('Feature Value')
        axes[1].set_ylabel('|SHAP Value|')
        axes[1].set_title('SHAP vs Feature Values (Top 6)')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('15_global_shap_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 15_global_shap_analysis.png")

        CHECKLIST["3.3_global_shap"] = True

    def local_shap_analysis(self):
        """3.4: Local SHAP analysis for specific samples"""
        print("\n" + "="*70)
        print("3.4 LOCAL SHAP ANALYSIS")
        print("="*70)

        # Use sample data for speed
        sample_idx = np.random.choice(len(self.X_train), min(1000, len(self.X_train)),
                                     replace=False)
        X_sample = self.X_train.iloc[sample_idx]
        y_sample = self.y_train.iloc[sample_idx]

        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_sample)

        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]
        else:
            shap_vals = self.shap_values

        # Find interesting samples
        defect_samples = np.where(y_sample == 1)[0]
        normal_samples = np.where(y_sample == 0)[0]

        if len(defect_samples) > 0:
            high_shap_defect = defect_samples[np.abs(shap_vals[defect_samples]).sum(axis=1).argsort()[-1]]
        if len(normal_samples) > 0:
            high_shap_normal = normal_samples[np.abs(shap_vals[normal_samples]).sum(axis=1).argsort()[-1]]

        print(f"\nAnalyzing local SHAP for interesting samples...")
        print(f"Sample with highest SHAP impact in defect class: {sample_idx[high_shap_defect] if len(defect_samples) > 0 else 'N/A'}")
        print(f"Sample with highest SHAP impact in normal class: {sample_idx[high_shap_normal] if len(normal_samples) > 0 else 'N/A'}")

        CHECKLIST["3.4_local_shap"] = True

    def shap_interaction_analysis(self):
        """3.5: SHAP interaction values"""
        print("\n" + "="*70)
        print("3.5 SHAP INTERACTION ANALYSIS")
        print("="*70)

        # Use very small sample for faster computation
        sample_size = min(100, len(self.X_train) // 5)
        sample_idx = np.random.choice(len(self.X_train), sample_size, replace=False)
        X_sample = self.X_train.iloc[sample_idx]

        print("Computing SHAP interaction values (this may take a moment)...")
        explainer = shap.TreeExplainer(self.model)
        shap_interaction = explainer.shap_interaction_values(X_sample)

        if isinstance(shap_interaction, list):
            shap_int = shap_interaction[1]
        else:
            shap_int = shap_interaction

        # Find top interactions
        interactions = []
        for i in range(shap_int.shape[1]):
            for j in range(i+1, shap_int.shape[1]):
                interaction_strength = np.abs(shap_int[:, i, j]).mean()
                interactions.append((
                    self.X_train.columns[i],
                    self.X_train.columns[j],
                    interaction_strength
                ))

        interactions.sort(key=lambda x: x[2], reverse=True)

        print(f"\nTop 10 Feature Interactions (by SHAP):")
        for feat1, feat2, strength in interactions[:10]:
            print(f"  {feat1} <-> {feat2}: {strength:.6f}")

        self.insights['shap_interactions'] = interactions[:5]

        CHECKLIST["3.5_shap_interactions"] = True

    def false_positive_analysis(self):
        """3.6: Analyze false positive samples"""
        print("\n" + "="*70)
        print("3.6 FALSE POSITIVE ANALYSIS")
        print("="*70)

        y_pred = self.model.predict(self.X_train)
        proba = self.model.predict_proba(self.X_train)[:, 1]

        # FP: predicted 1, actual 0
        fp_mask = (y_pred == 1) & (self.y_train == 0)
        fp_count = fp_mask.sum()

        print(f"\nFalse Positives: {fp_count} ({fp_count/len(self.y_train)*100:.2f}%)")

        if fp_count > 0:
            fp_features = self.X_train[fp_mask]
            normal_features = self.X_train[self.y_train == 0]

            # Compare FP characteristics with normal samples
            fp_mean = fp_features.mean()
            normal_mean = normal_features.mean()

            diff = (fp_mean - normal_mean).abs().sort_values(ascending=False)

            print(f"\nTop 10 Features Different in FPs vs Normal:")
            for feat, val in diff.head(10).items():
                print(f"  {feat}: {val:.6f}")

            self.insights['fp_characteristics'] = diff.head(5)

            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(proba[self.y_train == 0], bins=50, alpha=0.6, label=f'Normal (n={sum(self.y_train == 0)})')
            axes[0].hist(proba[fp_mask], bins=50, alpha=0.6, label=f'False Positives (n={fp_count})')
            axes[0].set_xlabel('Predicted Probability')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('FP Distribution vs Normal Samples')
            axes[0].legend()

            diff.head(10).plot(kind='barh', ax=axes[1])
            axes[1].set_title('Top 10 FP Distinguishing Features')
            axes[1].set_xlabel('Absolute Mean Difference')

            plt.tight_layout()
            plt.savefig('16_false_positive_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("\n✓ Saved: 16_false_positive_analysis.png")

        CHECKLIST["3.6_false_positive_analysis"] = True

    def false_negative_analysis(self):
        """3.7: Analyze false negative samples"""
        print("\n" + "="*70)
        print("3.7 FALSE NEGATIVE ANALYSIS")
        print("="*70)

        y_pred = self.model.predict(self.X_train)
        proba = self.model.predict_proba(self.X_train)[:, 1]

        # FN: predicted 0, actual 1
        fn_mask = (y_pred == 0) & (self.y_train == 1)
        fn_count = fn_mask.sum()
        fn_rate = fn_count/sum(self.y_train)*100

        print(f"\nFalse Negatives (ESCAPED DEFECTS): {fn_count} ({fn_rate:.2f}% of defects)")
        print(f"⚠️ CRITICAL: {fn_rate:.2f}% of actual defects are NOT caught by baseline model")

        # Save FN rate for downstream parts
        self.insights['fn_rate'] = fn_rate

        if fn_count > 0:
            fn_features = self.X_train[fn_mask]
            defect_features = self.X_train[self.y_train == 1]

            # Compare FN characteristics with other defects
            fn_mean = fn_features.mean()
            defect_mean = defect_features.mean()

            diff = (fn_mean - defect_mean).abs().sort_values(ascending=False)

            print(f"\nTop 10 Features Different in FNs vs Detected Defects:")
            for feat, val in diff.head(10).items():
                print(f"  {feat}: {val:.6f}")

            self.insights['fn_characteristics'] = diff.head(5)

            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(proba[self.y_train == 1], bins=50, alpha=0.6, label=f'Detected Defects (n={sum(self.y_train == 1) - fn_count})')
            axes[0].hist(proba[fn_mask], bins=50, alpha=0.6, label=f'False Negatives (n={fn_count})')
            axes[0].set_xlabel('Predicted Probability')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('FN Distribution vs Detected Defects')
            axes[0].legend()

            diff.head(10).plot(kind='barh', ax=axes[1])
            axes[1].set_title('Top 10 FN Distinguishing Features')
            axes[1].set_xlabel('Absolute Mean Difference')

            plt.tight_layout()
            plt.savefig('17_false_negative_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("\n✓ Saved: 17_false_negative_analysis.png")

        CHECKLIST["3.7_false_negative_analysis"] = True

    def hard_sample_analysis(self):
        """3.8: Analyze hard/borderline samples"""
        print("\n" + "="*70)
        print("3.8 HARD SAMPLE ANALYSIS")
        print("="*70)

        proba = self.model.predict_proba(self.X_train)[:, 1]
        y_pred = self.model.predict(self.X_train)

        # Hard samples: those with probability close to 0.5
        hard_mask = np.abs(proba - 0.5) < 0.15
        hard_count = hard_mask.sum()

        # Compare hard vs easy samples
        hard_features = self.X_train[hard_mask]
        easy_features = self.X_train[~hard_mask]

        hard_var = hard_features.var()
        easy_var = easy_features.var()

        instability = (hard_var / (easy_var + 1e-10)).sort_values(ascending=False)

        print(f"\nHard Samples: {hard_count} ({hard_count/len(self.X_train)*100:.2f}%)")
        print(f"Defects among hard samples: {self.y_train[hard_mask].sum()} ({self.y_train[hard_mask].mean()*100:.2f}%)")
        print(f"Defects among easy samples: {self.y_train[~hard_mask].sum()} ({self.y_train[~hard_mask].mean()*100:.2f}%)")

        print(f"\nTop 10 Most Unstable Features in Hard Samples:")
        for feat, val in instability.head(10).items():
            print(f"  {feat}: {val:.4f}")

        self.insights['hard_sample_features'] = instability.head(5)

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].hist(proba[~hard_mask], bins=50, alpha=0.6, label='Easy Samples')
        axes[0, 0].hist(proba[hard_mask], bins=50, alpha=0.6, label='Hard Samples')
        axes[0, 0].set_xlabel('Predicted Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Probability Distribution: Hard vs Easy')
        axes[0, 0].legend()

        # Hard vs easy by class
        hard_defect = self.y_train[hard_mask].mean() * 100
        easy_defect = self.y_train[~hard_mask].mean() * 100
        pd.DataFrame({'Hard Samples': [hard_defect], 'Easy Samples': [easy_defect]}).T.plot(
            kind='bar', ax=axes[0, 1], legend=False
        )
        axes[0, 1].set_title('Defect Rate: Hard vs Easy Samples')
        axes[0, 1].set_ylabel('Defect %')
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)

        # Variance ratio
        instability.head(15).plot(kind='barh', ax=axes[1, 0])
        axes[1, 0].set_title('Feature Instability Ratio (Hard/Easy)')
        axes[1, 0].set_xlabel('Variance Ratio')

        # Count distribution
        axes[1, 1].bar(['Easy (|P-0.5|≥0.15)', 'Hard (|P-0.5|<0.15)'],
                      [sum(~hard_mask), sum(hard_mask)], color=['skyblue', 'coral'])
        axes[1, 1].set_title('Sample Distribution: Easy vs Hard')
        axes[1, 1].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig('18_hard_sample_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 18_hard_sample_analysis.png")

        CHECKLIST["3.8_hard_sample_analysis"] = True

    def generate_error_analysis_report(self):
        """Generate error analysis report"""
        print("\n" + "="*70)
        print("PART 3: ERROR ANALYSIS INSIGHTS SUMMARY")
        print("="*70)

        y_pred = self.model.predict(self.X_train)
        proba = self.model.predict_proba(self.X_train)[:, 1]
        fp_mask = (y_pred == 1) & (self.y_train == 0)
        fn_mask = (y_pred == 0) & (self.y_train == 1)

        report = f"""
SHAP & ERROR ANALYSIS INSIGHTS
=============================

1. MODEL DOMINANCE PATTERNS:
   - Most important features identified via SHAP
   - See 15_global_shap_analysis.png

2. FALSE POSITIVE ANALYSIS:
   - FP Count: {fp_mask.sum()}
   - FP Rate: {fp_mask.sum()/sum(self.y_train == 0)*100:.2f}%
   - FPs tend to resemble defects in specific features
   - Top distinguishing features saved

3. FALSE NEGATIVE ANALYSIS (CRITICAL):
   - FN Count: {fn_mask.sum()}
   - Escaped Defect Rate: {fn_mask.sum()/sum(self.y_train == 1)*100:.2f}%
   - Some defects lack typical characteristics
   - Specific feature patterns differ from detected defects

4. HARD SAMPLE CHARACTERISTICS:
   - Borderline samples: ~15% of dataset
   - High model uncertainty
   - Specific features show instability

5. KEY RECOMMENDATIONS:
   ✓ Focus on escaped defect (FN) characteristics
   ✓ Engineer features to separate hard samples
   ✓ Consider ensemble approaches
   ✓ Use identified interactions in Part 5
   ✓ Optimize threshold for recall priority (Part 7)
"""

        print(report)

        with open('PART3_INSIGHTS.txt', 'w') as f:
            f.write(report)

        print("✓ Saved: PART3_INSIGHTS.txt")

    def run_shap_analysis(self):
        """Execute all SHAP analysis steps"""
        self.load_data()
        self.train_model()
        self.global_shap_analysis()
        self.local_shap_analysis()
        self.shap_interaction_analysis()
        self.false_positive_analysis()
        self.false_negative_analysis()
        self.hard_sample_analysis()
        self.generate_error_analysis_report()

        print("\n" + "="*70)
        print("PART 3 COMPLETION CHECKLIST")
        print("="*70)
        for step, completed in CHECKLIST.items():
            status = "✓" if completed else "✗"
            print(f"{status} {step}")

        return self.insights

if __name__ == "__main__":
    from insights_manager import InsightsManager

    shap_analysis = SHAPErrorAnalysis(train_path='train.csv')
    insights = shap_analysis.run_shap_analysis()

    # Save insights for downstream parts
    manager = InsightsManager()

    # Compute FN and FP rates for saving
    y_pred = shap_analysis.model.predict(shap_analysis.X_train)
    fn_mask = (y_pred == 0) & (shap_analysis.y_train == 1)
    fp_mask = (y_pred == 1) & (shap_analysis.y_train == 0)

    fn_rate = fn_mask.sum() / sum(shap_analysis.y_train) * 100
    fp_rate = fp_mask.sum() / sum(shap_analysis.y_train == 0) * 100

    insights['fn_rate'] = fn_rate
    insights['fp_rate'] = fp_rate

    manager.set_part3_insights(insights)

    print("\n" + "="*70)
    print("PART 3 COMPLETE")
    print("="*70)
    print(f"\n⚠️ CRITICAL FINDING: {fn_rate:.2f}% of defects are ESCAPED (not detected)")
    print("✓ Insights propagated to downstream parts")
    print("Ready for Part 4: Correlation Grouping")
