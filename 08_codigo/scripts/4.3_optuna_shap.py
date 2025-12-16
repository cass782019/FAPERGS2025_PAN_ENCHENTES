#!/usr/bin/env python3
"""
Otimização de Modelos com Optuna e Análise SHAP
Usa datasets gerados pelo pipeline 4.1

Este script:
- Carrega datasets processados do pipeline 4.1
- Testa múltiplos algoritmos de ML (XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting)
- Usa Optuna para otimização de hiperparâmetros
- Aplica SHAP para explicabilidade dos modelos
"""

import pandas as pd
import numpy as np
import warnings
import joblib
from pathlib import Path
from datetime import datetime
import logging

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)

# Modelos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost não disponível. Instale com: pip install catboost")

# Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# SHAP
try:
    import shap
    # shap.initjs()  # Removido para evitar problemas de renderização
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP não disponível. Instale com: pip install shap")

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configurações de diretórios
DATA_DIR = Path('./outputs/processed_data')
MODEL_DIR = Path('./outputs/models_optuna')
SHAP_DIR = Path('./outputs/shap_plots')

# Criar diretórios
MODEL_DIR.mkdir(exist_ok=True, parents=True)
SHAP_DIR.mkdir(exist_ok=True, parents=True)

print("✅ Bibliotecas importadas com sucesso!")


# ============================================================================
# 1. CARREGAMENTO DE DADOS
# ============================================================================

def carregar_datasets():
    """Carrega datasets processados do pipeline 4.1"""
    datasets = {}
    
    # Tentar carregar diferentes versões de datasets
    possiveis_arquivos = [
        'dataset_unificado.csv',
        'dataset_com_posts.csv',
        'dataset_sem_posts.csv'
    ]
    
    for arquivo in possiveis_arquivos:
        caminho = DATA_DIR / arquivo
        if caminho.exists():
            nome = arquivo.replace('.csv', '')
            datasets[nome] = pd.read_csv(caminho, low_memory=False)
            logger.info(f"✓ Carregado: {arquivo} - Shape: {datasets[nome].shape}")
    
    if not datasets:
        logger.warning("⚠️ Nenhum dataset encontrado. Execute o pipeline 4.1 primeiro.")
        logger.info(f"Procurando em: {DATA_DIR}")
    
    return datasets


# ============================================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================================

def preparar_dados(df, target_col, drop_features=None):
    """
    Prepara dados para treinamento
    
    Args:
        df: DataFrame
        target_col: Nome da coluna target
        drop_features: Lista de features para remover
    
    Returns:
        X, y, feature_names
    """
    if drop_features is None:
        drop_features = ['cnpj_basico_str', 'cnpj', 'profile_picture_url']
    
    # Remover features
    df_clean = df.drop(columns=[col for col in drop_features if col in df.columns], errors='ignore')
    
    # Separar target
    if target_col not in df_clean.columns:
        raise ValueError(f"Target '{target_col}' não encontrado no dataset")
    
    y = df_clean[target_col].copy()
    X = df_clean.drop(columns=[target_col], errors='ignore')
    
    # Remover outros targets se existirem
    outros_targets = ['sobreviveu_pandemia', 'sobreviveu_enchente']
    for t in outros_targets:
        if t != target_col and t in X.columns:
            X = X.drop(columns=[t])
    
    # Converter colunas categóricas para numéricas
    for col in X.select_dtypes(include=['object']).columns:
        if X[col].nunique() < 50:  # Se tiver poucos valores únicos, fazer encoding
            X[col] = pd.Categorical(X[col]).codes
        else:
            X = X.drop(columns=[col])  # Remover se tiver muitos valores únicos
    
    # Garantir que todas as colunas são numéricas
    X = X.select_dtypes(include=[np.number])
    
    logger.info(f"  Features: {X.shape[1]}, Amostras: {X.shape[0]}")
    logger.info(f"  Target '{target_col}': {y.sum()} positivos ({y.mean()*100:.2f}%)")
    
    return X, y, X.columns.tolist()


# ============================================================================
# 3. OTIMIZAÇÃO COM OPTUNA
# ============================================================================

class OtimizadorModelos:
    """Classe para otimização de múltiplos modelos com Optuna"""
    
    def __init__(self, X_train, y_train, X_val, y_val, n_trials=50):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_trials = n_trials
        self.best_models = {}
        self.studies = {}
    
    def objetivo_xgboost(self, trial):
        """Função objetivo para XGBoost"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'eval_metric': 'auc'
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_pred_proba)
        
        return auc
    
    def objetivo_lightgbm(self, trial):
        """Função objetivo para LightGBM"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_pred_proba)
        
        return auc
    
    def objetivo_catboost(self, trial):
        """Função objetivo para CatBoost"""
        if not CATBOOST_AVAILABLE:
            return 0.0
        
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'random_seed': 42,
            'verbose': False
        }
        
        model = cb.CatBoostClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_pred_proba)
        
        return auc
    
    def objetivo_random_forest(self, trial):
        """Função objetivo para Random Forest"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_pred_proba)
        
        return auc
    
    def objetivo_gradient_boosting(self, trial):
        """Função objetivo para Gradient Boosting"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
        
        model = GradientBoostingClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_pred_proba)
        
        return auc
    
    def otimizar_modelo(self, nome_modelo, n_trials=None):
        """Otimiza um modelo específico"""
        n_trials = n_trials or self.n_trials
        
        logger.info(f"\n🔍 Otimizando {nome_modelo}...")
        
        # Mapear nome para função objetivo
        objetivos = {
            'xgboost': self.objetivo_xgboost,
            'lightgbm': self.objetivo_lightgbm,
            'catboost': self.objetivo_catboost,
            'random_forest': self.objetivo_random_forest,
            'gradient_boosting': self.objetivo_gradient_boosting
        }
        
        if nome_modelo not in objetivos:
            raise ValueError(f"Modelo '{nome_modelo}' não suportado")
        
        # Criar study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Otimizar
        study.optimize(objetivos[nome_modelo], n_trials=n_trials, show_progress_bar=True)
        
        # Treinar melhor modelo
        best_params = study.best_params
        logger.info(f"  Melhor AUC: {study.best_value:.4f}")
        logger.info(f"  Melhores parâmetros: {best_params}")
        
        # Treinar modelo final com todos os dados de treino
        if nome_modelo == 'xgboost':
            best_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='auc')
        elif nome_modelo == 'lightgbm':
            best_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
        elif nome_modelo == 'catboost':
            best_model = cb.CatBoostClassifier(**best_params, random_seed=42, verbose=False)
        elif nome_modelo == 'random_forest':
            best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        elif nome_modelo == 'gradient_boosting':
            best_model = GradientBoostingClassifier(**best_params, random_state=42)
        
        best_model.fit(self.X_train, self.y_train)
        
        self.best_models[nome_modelo] = best_model
        self.studies[nome_modelo] = study
        
        return best_model, study
    
    def otimizar_todos(self, modelos=None):
        """Otimiza todos os modelos"""
        if modelos is None:
            modelos = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']
            if CATBOOST_AVAILABLE:
                modelos.append('catboost')
        
        resultados = {}
        
        for modelo in modelos:
            try:
                best_model, study = self.otimizar_modelo(modelo)
                resultados[modelo] = {
                    'model': best_model,
                    'study': study,
                    'best_auc': study.best_value
                }
            except Exception as e:
                logger.error(f"Erro ao otimizar {modelo}: {str(e)}")
        
        return resultados


# ============================================================================
# 4. ANÁLISE SHAP
# ============================================================================

class AnalisadorSHAP:
    """Classe para análise SHAP expandida dos modelos com múltiplos gráficos e estatísticas"""
    
    def __init__(self, model, X_train, X_test, feature_names, output_dir=None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.shap_explanation = None
        self.base_value = None
        
        # Diretório de saída
        if output_dir is None:
            self.output_dir = SHAP_DIR / 'shap2'
        else:
            self.output_dir = Path(output_dir) / 'shap2'
        
        # Criar subdiretórios
        self.subdirs = ['summary', 'individual', 'dependence', 'interaction', 
                       'partial_dependence', 'heatmaps', 'custom', 'violin', 'scatter']
        for subdir in self.subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def criar_explainer(self, sample_size=100):
        """Cria explainer SHAP"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP não está instalado. Instale com: pip install shap")
        
        logger.info("Criando explainer SHAP...")
        
        # Amostrar dados de treino para background (SHAP precisa de background)
        if len(self.X_train) > sample_size:
            background = self.X_train.sample(n=min(sample_size, len(self.X_train)), random_state=42)
        else:
            background = self.X_train
        
        # Criar explainer baseado no tipo de modelo
        model_type = type(self.model).__name__
        
        if 'XGB' in model_type or 'LGBM' in model_type or 'CatBoost' in model_type:
            # Tree-based models
            self.explainer = shap.TreeExplainer(self.model, background)
        else:
            # Outros modelos
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )
        
        logger.info("✓ Explainer criado")
        return self.explainer
    
    def calcular_shap_values(self, X=None, max_evals=100):
        """Calcula valores SHAP"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP não está instalado")
        
        if X is None:
            X = self.X_test
        
        if self.explainer is None:
            self.criar_explainer()
        
        logger.info(f"Calculando SHAP values para {len(X)} amostras...")
        
        # Para KernelExplainer, limitar amostras
        if isinstance(self.explainer, shap.KernelExplainer):
            if len(X) > max_evals:
                X = X.sample(n=max_evals, random_state=42)
            self.shap_values = self.explainer.shap_values(X, nsamples=max_evals)
        else:
            # TreeExplainer é rápido
            self.shap_values = self.explainer.shap_values(X)
        
        logger.info("✓ SHAP values calculados")
        return self.shap_values
    
    def plotar_importancia(self, save_path=None):
        """Plota importância das features"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP não está instalado")
        
        if self.shap_values is None:
            self.calcular_shap_values()
        
        # Ajustar para modelos binários (pegar apenas classe positiva)
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[1]  # Classe positiva
        else:
            shap_values_plot = self.shap_values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_plot,
            self.X_test.iloc[:len(shap_values_plot)] if hasattr(self.X_test, 'iloc') else self.X_test[:len(shap_values_plot)],
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Salvo: {save_path}")
        
        plt.close()  # Fechar figura ao invés de mostrar
    
    def plotar_importancia_bar(self, save_path=None):
        """Plota importância das features (bar plot)"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP não está instalado")
        
        if self.shap_values is None:
            self.calcular_shap_values()
        
        # Ajustar para modelos binários
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[1]
        else:
            shap_values_plot = self.shap_values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_plot,
            self.X_test.iloc[:len(shap_values_plot)] if hasattr(self.X_test, 'iloc') else self.X_test[:len(shap_values_plot)],
            feature_names=self.feature_names,
            plot_type='bar',
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Salvo: {save_path}")
        
        plt.close()  # Fechar figura ao invés de mostrar
    
    def obter_importancia_features(self, top_n=20):
        """Retorna importância das features"""
        if self.shap_values is None:
            self.calcular_shap_values()
        
        # Ajustar para modelos binários
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[1]
        else:
            shap_values_plot = self.shap_values
        
        # Calcular importância média (valor absoluto)
        importancia = np.abs(shap_values_plot).mean(axis=0)
        
        # Criar DataFrame
        df_importancia = pd.DataFrame({
            'feature': self.feature_names,
            'importancia': importancia
        }).sort_values('importancia', ascending=False)
        
        return df_importancia.head(top_n)
    
    def _preparar_shap_data(self):
        """Prepara dados SHAP para visualização"""
        if self.shap_values is None:
            self.calcular_shap_values()
        
        # Ajustar para modelos binários
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[1]
            if isinstance(self.explainer.expected_value, list):
                self.base_value = self.explainer.expected_value[1]
            else:
                self.base_value = self.explainer.expected_value
        else:
            shap_values_plot = self.shap_values
            self.base_value = self.explainer.expected_value
        
        # Converter para DataFrame se necessário
        if not isinstance(self.X_test, pd.DataFrame):
            X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        else:
            X_test_df = self.X_test.copy()
        
        # Criar Explanation object
        try:
            self.shap_explanation = shap.Explanation(
                values=shap_values_plot,
                base_values=np.full(len(X_test_df), self.base_value),
                data=X_test_df.values,
                feature_names=self.feature_names
            )
        except:
            # Fallback se Explanation não funcionar
            self.shap_explanation = None
        
        return shap_values_plot, X_test_df
    
    def gerar_analise_completa(self, modelo_nome="modelo", n_samples_shap=1000):
        """
        Gera análise SHAP completa com todos os gráficos e estatísticas
        
        Args:
            modelo_nome: Nome do modelo para prefixo dos arquivos
            n_samples_shap: Número de amostras para análise SHAP
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP não disponível. Pulando análise completa.")
            return
        
        logger.info("="*80)
        logger.info("GERANDO ANÁLISE SHAP EXPANDIDA")
        logger.info("="*80)
        
        # Preparar dados
        shap_values_plot, X_test_df = self._preparar_shap_data()
        
        # Limitar amostras se necessário
        if len(X_test_df) > n_samples_shap:
            sample_idx = np.random.choice(len(X_test_df), n_samples_shap, replace=False)
            X_test_sample = X_test_df.iloc[sample_idx].copy()
            shap_values_sample = shap_values_plot[sample_idx]
        else:
            X_test_sample = X_test_df.copy()
            shap_values_sample = shap_values_plot
        
        # Calcular importância
        shap_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': np.abs(shap_values_sample).mean(0)
        }).sort_values('shap_importance', ascending=False)
        
        top_features = shap_importance.head(10)['feature'].values.tolist()
        top_5_features = shap_importance.head(5)['feature'].values.tolist()
        
        logger.info(f"\n📁 Diretório de saída: {self.output_dir}")
        logger.info(f"📊 Gerando gráficos para {len(X_test_sample)} amostras...")
        
        # ========================================================================
        # 1. SUMMARY PLOTS
        # ========================================================================
        logger.info("\n1️⃣ Gerando Summary Plots...")
        self._gerar_summary_plots(shap_values_sample, X_test_sample, modelo_nome)
        
        # ========================================================================
        # 2. INDIVIDUAL PREDICTION PLOTS
        # ========================================================================
        logger.info("\n2️⃣ Gerando Individual Prediction Plots...")
        self._gerar_individual_plots(shap_values_sample, X_test_sample, modelo_nome)
        
        # ========================================================================
        # 3. DEPENDENCE PLOTS
        # ========================================================================
        logger.info("\n3️⃣ Gerando Dependence Plots...")
        self._gerar_dependence_plots(shap_values_sample, X_test_sample, top_features, modelo_nome)
        
        # ========================================================================
        # 4. INTERACTION PLOTS
        # ========================================================================
        logger.info("\n4️⃣ Gerando Interaction Plots...")
        self._gerar_interaction_plots(top_5_features, modelo_nome)
        
        # ========================================================================
        # 5. PARTIAL DEPENDENCE PLOTS
        # ========================================================================
        logger.info("\n5️⃣ Gerando Partial Dependence Plots...")
        self._gerar_partial_dependence_plots(top_5_features, X_test_sample, modelo_nome)
        
        # ========================================================================
        # 6. HEATMAPS
        # ========================================================================
        logger.info("\n6️⃣ Gerando Heatmaps...")
        self._gerar_heatmaps(shap_values_sample, top_features, modelo_nome)
        
        # ========================================================================
        # 7. VIOLIN PLOTS
        # ========================================================================
        logger.info("\n7️⃣ Gerando Violin Plots...")
        self._gerar_violin_plots(top_5_features, modelo_nome)
        
        # ========================================================================
        # 8. SCATTER PLOTS
        # ========================================================================
        logger.info("\n8️⃣ Gerando Scatter Plots...")
        self._gerar_scatter_plots(top_features, modelo_nome)
        
        # ========================================================================
        # 9. CUSTOM PLOTS
        # ========================================================================
        logger.info("\n9️⃣ Gerando Custom Plots...")
        self._gerar_custom_plots(shap_values_sample, shap_importance, top_features, X_test_sample, modelo_nome)
        
        # ========================================================================
        # 10. SALVAR ESTATÍSTICAS
        # ========================================================================
        logger.info("\n🔟 Salvando estatísticas...")
        self._salvar_estatisticas(shap_values_sample, shap_importance, modelo_nome)
        
        logger.info("\n" + "="*80)
        logger.info("✅ ANÁLISE SHAP COMPLETA CONCLUÍDA!")
        logger.info("="*80)
        logger.info(f"📁 Todos os arquivos salvos em: {self.output_dir.absolute()}")
    
    def _gerar_summary_plots(self, shap_values, X_test, modelo_nome):
        """Gera summary plots adicionais"""
        try:
            # Beeswarm Top 30
            plt.figure(figsize=(14, 12))
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, 
                            show=False, max_display=30)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'summary' / f'{modelo_nome}_beeswarm_top30.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("   ✅ Summary plot (beeswarm) - Top 30")
            
            # Bar todas features
            plt.figure(figsize=(12, max(10, len(self.feature_names) * 0.3)))
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, 
                            plot_type="bar", show=False, max_display=len(self.feature_names))
            plt.tight_layout()
            plt.savefig(self.output_dir / 'summary' / f'{modelo_nome}_bar_all.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("   ✅ Summary plot (bar) - Todas features")
            
            # Violin Top 15
            if self.shap_explanation is not None:
                try:
                    plt.figure(figsize=(14, 10))
                    shap.plots.violin(self.shap_explanation, show=False, max_display=15)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'summary' / f'{modelo_nome}_violin_top15.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("   ✅ Summary plot (violin) - Top 15")
                except:
                    logger.warning("   ⚠️ Violin plot não disponível")
        except Exception as e:
            logger.error(f"   ❌ Erro em summary plots: {e}")
    
    def _gerar_individual_plots(self, shap_values, X_test, modelo_nome):
        """Gera plots individuais (waterfall, force)"""
        try:
            n_waterfall = min(10, len(X_test))
            for i in range(n_waterfall):
                try:
                    plt.figure(figsize=(12, 8))
                    shap.waterfall_plot(shap.Explanation(
                        values=shap_values[i],
                        base_values=self.base_value,
                        data=X_test.iloc[i].values,
                        feature_names=self.feature_names
                    ), show=False, max_display=20)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'individual' / f'{modelo_nome}_waterfall_{i+1}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.warning(f"   ⚠️ Erro no waterfall {i+1}: {e}")
            logger.info(f"   ✅ Waterfall plots - {n_waterfall} amostras")
        except Exception as e:
            logger.error(f"   ❌ Erro em individual plots: {e}")
    
    def _gerar_dependence_plots(self, shap_values, X_test, top_features, modelo_nome):
        """Gera dependence plots"""
        try:
            # Individuais
            for feature in top_features:
                try:
                    feature_idx = self.feature_names.index(feature)
                    plt.figure(figsize=(10, 8))
                    shap.dependence_plot(feature_idx, shap_values, X_test, 
                                        feature_names=self.feature_names, show=False)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'dependence' / f'{modelo_nome}_dependence_{feature}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.warning(f"   ⚠️ Erro no dependence plot {feature}: {e}")
            logger.info(f"   ✅ Dependence plots individuais - {len(top_features)} features")
            
            # Grid Top 6
            fig, axes = plt.subplots(2, 3, figsize=(20, 14))
            axes = axes.flatten()
            for idx, feature in enumerate(top_features[:6]):
                try:
                    feature_idx = self.feature_names.index(feature)
                    shap.dependence_plot(feature_idx, shap_values, X_test, 
                                        feature_names=self.feature_names, ax=axes[idx], show=False)
                    axes[idx].set_title(f'{feature}', fontweight='bold', fontsize=11)
                except:
                    axes[idx].text(0.5, 0.5, f'Erro: {feature}', ha='center', va='center',
                                 transform=axes[idx].transAxes)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'dependence' / f'{modelo_nome}_dependence_grid.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("   ✅ Dependence plots em grid - Top 6")
        except Exception as e:
            logger.error(f"   ❌ Erro em dependence plots: {e}")
    
    def _gerar_interaction_plots(self, top_5_features, modelo_nome):
        """Gera interaction plots"""
        try:
            if self.shap_explanation is None:
                logger.warning("   ⚠️ Interaction plots requerem Explanation object")
                return
            
            for i, feature1 in enumerate(top_5_features[:3]):
                for feature2 in top_5_features[i+1:4] if i < 2 else []:
                    try:
                        feature1_idx = self.feature_names.index(feature1)
                        feature2_idx = self.feature_names.index(feature2)
                        plt.figure(figsize=(12, 8))
                        shap.plots.scatter(self.shap_explanation[:, feature1_idx], 
                                         color=self.shap_explanation[:, feature2_idx],
                                         show=False)
                        plt.title(f'Interaction: {feature1} vs {feature2}', fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(self.output_dir / 'interaction' / 
                                   f'{modelo_nome}_interaction_{feature1}_vs_{feature2}.png', 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        logger.warning(f"   ⚠️ Erro no interaction plot {feature1} vs {feature2}: {e}")
            logger.info("   ✅ Interaction plots gerados")
        except Exception as e:
            logger.error(f"   ❌ Erro em interaction plots: {e}")
    
    def _gerar_partial_dependence_plots(self, top_5_features, X_test, modelo_nome):
        """Gera partial dependence plots"""
        try:
            for feature in top_5_features:
                try:
                    feature_idx = self.feature_names.index(feature)
                    plt.figure(figsize=(10, 8))
                    shap.plots.partial_dependence(feature_idx, self.model.predict, 
                                                X_test, ice=False, 
                                                model_expected_value=True,
                                                feature_expected_value=True,
                                                show=False)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'partial_dependence' / f'{modelo_nome}_pdp_{feature}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.warning(f"   ⚠️ Erro no PDP {feature}: {e}")
            logger.info("   ✅ Partial Dependence Plots - Top 5")
        except Exception as e:
            logger.error(f"   ❌ Erro em partial dependence plots: {e}")
    
    def _gerar_heatmaps(self, shap_values, top_features, modelo_nome):
        """Gera heatmaps"""
        try:
            if self.shap_explanation is not None:
                # Heatmap SHAP values
                try:
                    plt.figure(figsize=(14, 10))
                    shap.plots.heatmap(self.shap_explanation[:, top_features[:20]], show=False)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'heatmaps' / f'{modelo_nome}_heatmap_top20.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("   ✅ Heatmap - Top 20 features")
                except:
                    logger.warning("   ⚠️ Heatmap SHAP não disponível")
            
            # Heatmap correlação
            try:
                shap_df = pd.DataFrame(shap_values[:, :len(top_features)], columns=top_features)
                corr_matrix = shap_df.corr()
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, linewidths=0.5)
                plt.title('Correlação entre SHAP Values (Top Features)', fontweight='bold', fontsize=14)
                plt.tight_layout()
                plt.savefig(self.output_dir / 'heatmaps' / f'{modelo_nome}_correlation_heatmap.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("   ✅ Heatmap de correlação SHAP")
            except Exception as e:
                logger.warning(f"   ⚠️ Erro no heatmap de correlação: {e}")
        except Exception as e:
            logger.error(f"   ❌ Erro em heatmaps: {e}")
    
    def _gerar_violin_plots(self, top_5_features, modelo_nome):
        """Gera violin plots"""
        try:
            if self.shap_explanation is None:
                logger.warning("   ⚠️ Violin plots requerem Explanation object")
                return
            
            for feature in top_5_features:
                try:
                    feature_idx = self.feature_names.index(feature)
                    plt.figure(figsize=(10, 8))
                    shap.plots.violin(self.shap_explanation[:, feature_idx], show=False)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'violin' / f'{modelo_nome}_violin_{feature}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.warning(f"   ⚠️ Erro no violin plot {feature}: {e}")
            logger.info("   ✅ Violin plots - Top 5")
        except Exception as e:
            logger.error(f"   ❌ Erro em violin plots: {e}")
    
    def _gerar_scatter_plots(self, top_features, modelo_nome):
        """Gera scatter plots"""
        try:
            if self.shap_explanation is None:
                logger.warning("   ⚠️ Scatter plots requerem Explanation object")
                return
            
            for feature in top_features:
                try:
                    feature_idx = self.feature_names.index(feature)
                    plt.figure(figsize=(10, 8))
                    shap.plots.scatter(self.shap_explanation[:, feature_idx], show=False)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'scatter' / f'{modelo_nome}_scatter_{feature}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.warning(f"   ⚠️ Erro no scatter plot {feature}: {e}")
            logger.info("   ✅ Scatter plots - Top 10")
        except Exception as e:
            logger.error(f"   ❌ Erro em scatter plots: {e}")
    
    def _gerar_custom_plots(self, shap_values, shap_importance, top_features, X_test, modelo_nome):
        """Gera gráficos customizados"""
        try:
            # Feature Importance Top 30
            plt.figure(figsize=(12, 14))
            top_30 = shap_importance.head(30)
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_30)))
            plt.barh(range(len(top_30)), top_30['shap_importance'].values, color=colors, alpha=0.8)
            plt.yticks(range(len(top_30)), top_30['feature'].values, fontsize=9)
            plt.xlabel('SHAP Importance', fontsize=12, fontweight='bold')
            plt.title('Top 30 Features - SHAP Importance', fontweight='bold', fontsize=16)
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'custom' / f'{modelo_nome}_importance_top30.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("   ✅ Feature Importance - Top 30")
            
            # Distribuição Top 10
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            for idx, feature in enumerate(top_features[:10]):
                feature_idx = self.feature_names.index(feature)
                axes[idx].hist(shap_values[:, feature_idx], bins=50, alpha=0.7, 
                              color='steelblue', edgecolor='black')
                axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
                axes[idx].set_title(f'{feature}', fontweight='bold', fontsize=9)
                axes[idx].set_xlabel('SHAP Value', fontsize=8)
                axes[idx].set_ylabel('Frequency', fontsize=8)
                axes[idx].grid(True, alpha=0.3)
            plt.suptitle('Distribuição de SHAP Values - Top 10 Features', 
                        fontweight='bold', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'custom' / f'{modelo_nome}_distribution_top10.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("   ✅ Distribuição SHAP values - Top 10")
            
            # Box plots Top 15
            shap_df_top15 = pd.DataFrame(shap_values[:, :len(top_features[:15])], 
                                        columns=top_features[:15])
            plt.figure(figsize=(14, 8))
            shap_df_top15.boxplot(rot=45, fontsize=9)
            plt.title('Box Plot de SHAP Values - Top 15 Features', fontweight='bold', fontsize=14)
            plt.ylabel('SHAP Value', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'custom' / f'{modelo_nome}_boxplot_top15.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("   ✅ Box plots SHAP values - Top 15")
            
            # Matriz SHAP values
            plt.figure(figsize=(14, max(8, len(X_test) * 0.1)))
            shap_matrix = shap_values[:, [self.feature_names.index(f) for f in top_features[:20]]]
            plt.imshow(shap_matrix, aspect='auto', cmap='RdBu_r', 
                      vmin=-np.abs(shap_matrix).max(), vmax=np.abs(shap_matrix).max())
            plt.colorbar(label='SHAP Value')
            plt.yticks(range(0, len(X_test), max(1, len(X_test)//10)), 
                      range(0, len(X_test), max(1, len(X_test)//10)))
            plt.xticks(range(len(top_features[:20])), top_features[:20], rotation=45, ha='right')
            plt.xlabel('Features', fontsize=12, fontweight='bold')
            plt.ylabel('Amostras', fontsize=12, fontweight='bold')
            plt.title('Matriz de SHAP Values (Amostras x Features) - Top 20', 
                     fontweight='bold', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'custom' / f'{modelo_nome}_matrix_top20.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("   ✅ Matriz SHAP values - Top 20")
        except Exception as e:
            logger.error(f"   ❌ Erro em custom plots: {e}")
    
    def _salvar_estatisticas(self, shap_values, shap_importance, modelo_nome):
        """Salva estatísticas e dados"""
        try:
            # Importância completa
            shap_importance.to_csv(self.output_dir / f'{modelo_nome}_importance_completo.csv', index=False)
            logger.info("   ✅ Importância SHAP completa salva")
            
            # SHAP values CSV
            shap_values_df = pd.DataFrame(shap_values, columns=self.feature_names)
            shap_values_df.to_csv(self.output_dir / f'{modelo_nome}_shap_values.csv', index=False)
            logger.info("   ✅ SHAP values salvos (CSV)")
            
            # Estatísticas descritivas
            shap_stats = pd.DataFrame({
                'feature': self.feature_names,
                'mean_shap': shap_values.mean(0),
                'std_shap': shap_values.std(0),
                'min_shap': shap_values.min(0),
                'max_shap': shap_values.max(0),
                'abs_mean_shap': np.abs(shap_values).mean(0),
                'positive_pct': (shap_values > 0).mean(0) * 100,
                'negative_pct': (shap_values < 0).mean(0) * 100
            })
            shap_stats = shap_stats.sort_values('abs_mean_shap', ascending=False)
            shap_stats.to_csv(self.output_dir / f'{modelo_nome}_statistics.csv', index=False)
            logger.info("   ✅ Estatísticas SHAP salvas")
            
            # Resumo em texto
            with open(self.output_dir / f'{modelo_nome}_resumo.txt', 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("RESUMO DA ANÁLISE SHAP EXPANDIDA\n")
                f.write("="*80 + "\n\n")
                f.write(f"Modelo: {modelo_nome}\n")
                f.write(f"Número de amostras: {len(shap_values)}\n")
                f.write(f"Número de features: {len(self.feature_names)}\n")
                f.write(f"Base value: {self.base_value:.4f}\n\n")
                f.write("Top 10 Features por Importância SHAP:\n")
                f.write("-"*80 + "\n")
                for idx, row in shap_importance.head(10).iterrows():
                    f.write(f"{idx+1:2d}. {row['feature']:30s} - {row['shap_importance']:.6f}\n")
                f.write("\n" + "="*80 + "\n")
            logger.info("   ✅ Resumo da análise salvo")
        except Exception as e:
            logger.error(f"   ❌ Erro ao salvar estatísticas: {e}")


# ============================================================================
# 5. PIPELINE PRINCIPAL
# ============================================================================

def executar_pipeline_completo(dataset_name, target_col, n_trials=50, modelos=None):
    """
    Executa pipeline completo: otimização + SHAP
    
    Args:
        dataset_name: Nome do dataset (chave do dict datasets)
        target_col: Nome da coluna target
        n_trials: Número de trials para Optuna
        modelos: Lista de modelos para otimizar (None = todos)
    """
    logger.info("=" * 80)
    logger.info(f"PIPELINE COMPLETO: {dataset_name} - {target_col}")
    logger.info("=" * 80)
    
    # 1. Carregar e preparar dados
    df = datasets[dataset_name]
    X, y, feature_names = preparar_dados(df, target_col)
    
    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Split adicional para validação (usado no Optuna)
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 3. Imputação e scaling
    imputer = SimpleImputer(strategy='mean')
    X_train_opt = pd.DataFrame(
        imputer.fit_transform(X_train_opt),
        columns=feature_names,
        index=X_train_opt.index
    )
    X_val_opt = pd.DataFrame(
        imputer.transform(X_val_opt),
        columns=feature_names,
        index=X_val_opt.index
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=feature_names,
        index=X_test.index
    )
    
    scaler = StandardScaler()
    X_train_opt_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_opt),
        columns=feature_names,
        index=X_train_opt.index
    )
    X_val_opt_scaled = pd.DataFrame(
        scaler.transform(X_val_opt),
        columns=feature_names,
        index=X_val_opt.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index
    )
    
    # 4. Otimização com Optuna
    otimizador = OtimizadorModelos(
        X_train_opt_scaled, y_train_opt,
        X_val_opt_scaled, y_val_opt,
        n_trials=n_trials
    )
    
    resultados = otimizador.otimizar_todos(modelos=modelos)
    
    # 5. Avaliar modelos no conjunto de teste
    logger.info("\n📊 Avaliando modelos no conjunto de teste...")
    resultados_finais = {}
    
    for nome_modelo, resultado in resultados.items():
        modelo = resultado['model']
        
        # Predições
        y_pred = modelo.predict(X_test_scaled)
        y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]
        
        # Métricas
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        
        resultados_finais[nome_modelo] = {
            'model': modelo,
            'auc': auc,
            'ap': ap,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info(f"  {nome_modelo}: AUC={auc:.4f}, AP={ap:.4f}")
    
    # 6. Selecionar melhor modelo
    melhor_modelo_nome = max(resultados_finais, key=lambda x: resultados_finais[x]['auc'])
    melhor_modelo = resultados_finais[melhor_modelo_nome]
    
    logger.info(f"\n🏆 Melhor modelo: {melhor_modelo_nome} (AUC={melhor_modelo['auc']:.4f})")
    
    # 7. Análise SHAP do melhor modelo
    if SHAP_AVAILABLE:
        logger.info("\n🔍 Executando análise SHAP expandida...")
        
        # Treinar modelo final com todos os dados de treino
        X_train_final = pd.concat([X_train_opt_scaled, X_val_opt_scaled])
        y_train_final = pd.concat([y_train_opt, y_val_opt])
        
        melhor_modelo['model'].fit(X_train_final, y_train_final)
        
        # Criar analisador com diretório customizado
        save_prefix = f"{dataset_name}_{target_col}_{melhor_modelo_nome}"
        analisador = AnalisadorSHAP(
            melhor_modelo['model'],
            X_train_final,
            X_test_scaled,
            feature_names,
            output_dir=SHAP_DIR  # Usar SHAP_DIR como base
        )
        
        # Gerar gráficos básicos (mantendo compatibilidade)
        analisador.plotar_importancia_bar(
            save_path=SHAP_DIR / f"{save_prefix}_importance_bar.png"
        )
        
        analisador.plotar_importancia(
            save_path=SHAP_DIR / f"{save_prefix}_importance_summary.png"
        )
        
        # Gerar análise completa expandida
        logger.info("\n📊 Gerando análise SHAP expandida...")
        analisador.gerar_analise_completa(
            modelo_nome=save_prefix,
            n_samples_shap=1000
        )
        
        # Obter importância das features
        importancia_df = analisador.obter_importancia_features(top_n=20)
        print("\n📈 Top 20 Features mais importantes:")
        print(importancia_df.to_string(index=False))
    else:
        analisador = None
        importancia_df = pd.DataFrame()
        logger.warning("SHAP não disponível. Pulando análise de explicabilidade.")
    
    # 8. Salvar modelo e resultados
    modelo_save_path = MODEL_DIR / f"best_{dataset_name}_{target_col}_{melhor_modelo_nome}.joblib"
    joblib.dump({
        'model': melhor_modelo['model'],
        'scaler': scaler,
        'imputer': imputer,
        'feature_names': feature_names,
        'auc': melhor_modelo['auc'],
        'ap': melhor_modelo['ap'],
        'importancia_features': importancia_df if SHAP_AVAILABLE else None
    }, modelo_save_path)
    
    logger.info(f"\n💾 Modelo salvo: {modelo_save_path}")
    
    return {
        'resultados': resultados_finais,
        'melhor_modelo': melhor_modelo_nome,
        'analisador_shap': analisador,
        'importancia_features': importancia_df
    }


# ============================================================================
# 6. COMPARAÇÃO DE MODELOS
# ============================================================================

def comparar_modelos(resultados):
    """Compara performance de todos os modelos"""
    comparacao = []
    
    for chave, resultado in resultados.items():
        for modelo_nome, modelo_info in resultado['resultados'].items():
            comparacao.append({
                'dataset_target': chave,
                'modelo': modelo_nome,
                'auc': modelo_info['auc'],
                'ap': modelo_info['ap']
            })
    
    df_comparacao = pd.DataFrame(comparacao)
    
    # Ordenar por AUC
    df_comparacao = df_comparacao.sort_values('auc', ascending=False)
    
    print("\n📊 Comparação de Modelos:")
    print(df_comparacao.to_string(index=False))
    
    # Plotar comparação
    if len(df_comparacao) > 0:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=df_comparacao, x='modelo', y='auc', hue='dataset_target')
        plt.title('AUC-ROC por Modelo')
        plt.xticks(rotation=45)
        plt.ylabel('AUC-ROC')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=df_comparacao, x='modelo', y='ap', hue='dataset_target')
        plt.title('Average Precision por Modelo')
        plt.xticks(rotation=45)
        plt.ylabel('Average Precision')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(SHAP_DIR / 'comparacao_modelos.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return df_comparacao


# ============================================================================
# 7. EXECUÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Carregar datasets
    datasets = carregar_datasets()
    
    if not datasets:
        print("⚠️ Nenhum dataset encontrado. Execute o pipeline 4.1 primeiro.")
        print(f"Procurando em: {DATA_DIR}")
    else:
        todos_resultados = {}
        
        # Escolher primeiro dataset disponível (ou especificar qual usar)
        dataset_escolhido = list(datasets.keys())[0]
        
        # Executar para cada target
        targets = ['sobreviveu_pandemia', 'sobreviveu_enchente']
        
        for target in targets:
            if target in datasets[dataset_escolhido].columns:
                print(f"\n{'='*80}")
                print(f"Executando para: {dataset_escolhido} - {target}")
                print(f"{'='*80}")
                
                resultado = executar_pipeline_completo(
                    dataset_name=dataset_escolhido,
                    target_col=target,
                    n_trials=50,  # Ajustar conforme necessário
                    modelos=['xgboost', 'lightgbm', 'random_forest']  # Modelos a testar
                )
                
                todos_resultados[f"{dataset_escolhido}_{target}"] = resultado
            else:
                print(f"⚠️ Target '{target}' não encontrado em {dataset_escolhido}")
        
        # Comparar modelos
        if todos_resultados:
            df_comparacao = comparar_modelos(todos_resultados)
            print("\n✅ Pipeline concluído com sucesso!")

