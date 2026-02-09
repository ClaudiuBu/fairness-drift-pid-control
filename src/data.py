# data.py
import numpy as np
from folktables import ACSDataSource, ACSIncome, ACSEmployment
from sklearn.preprocessing import StandardScaler


# ============================================================================
# SYNTHETIC DATA (ORIGINAL)
# ============================================================================
def generate_batch(n=500, drift_strength=0.0):
    """Generate synthetic batch with controlled fairness drift."""
    # Protejat vs Neprotejat (50/50 e mai stabil pentru demo)
    A = np.random.randint(0, 2, n) 
    
    # X corelat cu A
    X = np.random.randn(n, 2)
    X[:, 0] += A * 0.5 

    # Drift-ul afectează direct logica de decizie (discriminare "slow-creep")
    # drift_strength mare = penalizare mare pentru grupul A=0
    bias_against_0 = drift_strength 
    
    logits = X[:, 0] + 0.5 * X[:, 1] - bias_against_0 * (1 - A)
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    return X, y, A


# ============================================================================
# FOLKTABLES DATA (REAL-WORLD)
# ============================================================================
class FolktablesDataStream:
    """
    Wrapper for streaming real-world data from Folktables.
    
    Simulates temporal drift by using data from different years or states.
    
    Modes:
    - 'temporal': Uses multiple years sequentially (real temporal drift)
    - 'static': Uses single year, shuffled batches (no drift)
    - 'geographic': Uses different states sequentially (geographic shift)
    """
    
    def __init__(self, task='income', states=['CA'], years=[2018], 
                 sensitive_attribute='SEX', batch_size=500, mode='static'):
        """
        Args:
            task: 'income' or 'employment'
            states: list of state codes (e.g., ['CA', 'NY', 'TX'])
            years: list of years (e.g., [2014, 2015, 2016, 2017, 2018])
            sensitive_attribute: 'SEX', 'RAC1P' (race), or 'AGEP' (age)
            batch_size: number of samples per batch
            mode: 'temporal' (multi-year drift), 'static' (single year), 
                  or 'geographic' (multi-state drift)
        """
        self.task_name = task
        self.states = states
        self.years = sorted(years)  # Ensure chronological order
        self.sensitive_attribute = sensitive_attribute
        self.batch_size = batch_size
        self.mode = mode
        
        # Select task
        self.task = ACSIncome if task == 'income' else ACSEmployment
        
        # Prepare data based on mode
        self._prepare_data()
        
    def _prepare_data(self):
        """Load and preprocess data based on mode."""
        if self.mode == 'temporal':
            self._prepare_temporal_data()
        elif self.mode == 'geographic':
            self._prepare_geographic_data()
        else:  # static
            self._prepare_static_data()
    
    def _prepare_temporal_data(self):
        """Load data from multiple years to simulate temporal drift."""
        self.year_datasets = []
        self.scaler = None
        
        for year in self.years:
            data_source = ACSDataSource(survey_year=str(year), horizon='1-Year', 
                                       survey='person')
            
            all_X, all_y, all_A = [], [], []
            for state in self.states:
                acs_data = data_source.get_data(states=[state], download=True)
                X, y, _ = self.task.df_to_pandas(acs_data)
                
                # Extract sensitive attribute
                A = self._extract_sensitive_attribute(X)
                
                # Drop sensitive attribute from features
                X = X.drop(columns=[self.sensitive_attribute], errors='ignore')
                
                all_X.append(X.values)
                all_y.append(y.values)
                all_A.append(A.values)
            
            # Concatenate data for this year
            X_year = np.vstack(all_X)
            y_year = np.concatenate(all_y).ravel()
            A_year = np.concatenate(all_A)
            
            # Fit scaler on first year, transform all
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_year = self.scaler.fit_transform(X_year)
            else:
                X_year = self.scaler.transform(X_year)
            
            self.year_datasets.append({
                'X': X_year, 'y': y_year, 'A': A_year, 'year': year
            })
        
        # Start with first year
        self.current_year_idx = 0
        self.current_batch_idx = 0
        self._update_current_year_data()
    
    def _prepare_geographic_data(self):
        """Load data from multiple states to simulate geographic drift."""
        self.state_datasets = []
        self.scaler = None
        
        # Use first year only
        data_source = ACSDataSource(survey_year=str(self.years[0]), 
                                   horizon='1-Year', survey='person')
        
        for state in self.states:
            acs_data = data_source.get_data(states=[state], download=True)
            X, y, _ = self.task.df_to_pandas(acs_data)
            
            # Extract sensitive attribute
            A = self._extract_sensitive_attribute(X)
            
            # Drop sensitive attribute from features
            X = X.drop(columns=[self.sensitive_attribute], errors='ignore')
            
            # Standardize
            if self.scaler is None:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X.values)
            else:
                X = self.scaler.transform(X.values)
            
            self.state_datasets.append({
                'X': X, 'y': y.values.ravel(), 'A': A.values, 'state': state
            })
        
        # Start with first state
        self.current_state_idx = 0
        self.current_batch_idx = 0
        self._update_current_state_data()
    
    def _prepare_static_data(self):
        """Load data from single year/state (original behavior)."""
        data_source = ACSDataSource(survey_year=str(self.years[0]), 
                                   horizon='1-Year', survey='person')
        
        all_X, all_y, all_A = [], [], []
        
        for state in self.states:
            acs_data = data_source.get_data(states=[state], download=True)
            X, y, _ = self.task.df_to_pandas(acs_data)
            
            # Extract sensitive attribute
            A = self._extract_sensitive_attribute(X)
            
            # Drop sensitive attribute from features
            X = X.drop(columns=[self.sensitive_attribute], errors='ignore')
            
            all_X.append(X.values)
            all_y.append(y.values)
            all_A.append(A.values)
        
        # Concatenate all data
        self.X_all = np.vstack(all_X)
        self.y_all = np.concatenate(all_y).ravel()
        self.A_all = np.concatenate(all_A)
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_all = self.scaler.fit_transform(self.X_all)
        
        # Shuffle for i.i.d. batches
        perm = np.random.permutation(len(self.y_all))
        self.X_all = self.X_all[perm]
        self.y_all = self.y_all[perm]
        self.A_all = self.A_all[perm]
        
        # Initialize batch counter
        self.current_idx = 0
        self.total_samples = len(self.y_all)
    
    def _extract_sensitive_attribute(self, X):
        """Extract and binarize sensitive attribute."""
        if self.sensitive_attribute == 'SEX':
            return (X['SEX'] == 1).astype(int)  # 1=Male, 0=Female
        elif self.sensitive_attribute == 'RAC1P':
            return (X['RAC1P'] == 1).astype(int)  # 1=White, 0=Non-White
        else:
            # For age or other continuous attributes, binarize at median
            return (X[self.sensitive_attribute] > X[self.sensitive_attribute].median()).astype(int)
    
    def _update_current_year_data(self):
        """Update pointers for current year in temporal mode."""
        dataset = self.year_datasets[self.current_year_idx]
        self.X_current = dataset['X']
        self.y_current = dataset['y']
        self.A_current = dataset['A']
        self.total_samples_current = len(self.y_current)
    
    def _update_current_state_data(self):
        """Update pointers for current state in geographic mode."""
        dataset = self.state_datasets[self.current_state_idx]
        self.X_current = dataset['X']
        self.y_current = dataset['y']
        self.A_current = dataset['A']
        self.total_samples_current = len(self.y_current)
        
    def get_batch(self, drift_factor=0.0):
        """
        Get next batch from the stream.
        
        In temporal/geographic modes, automatically progresses through years/states
        to simulate drift.
        
        Args:
            drift_factor: optional parameter for compatibility with synthetic data
                         In temporal mode, can be used to control year progression
        
        Returns:
            X, y, A: features, labels, sensitive attributes
        """
        if self.mode == 'temporal':
            return self._get_batch_temporal()
        elif self.mode == 'geographic':
            return self._get_batch_geographic()
        else:  # static
            return self._get_batch_static()
    
    def _get_batch_temporal(self):
        """Get batch in temporal mode with year progression."""
        start_idx = self.current_batch_idx
        end_idx = min(start_idx + self.batch_size, self.total_samples_current)
        
        X = self.X_current[start_idx:end_idx]
        y = self.y_current[start_idx:end_idx]
        A = self.A_current[start_idx:end_idx]
        
        self.current_batch_idx = end_idx
        
        # If finished current year, move to next
        if self.current_batch_idx >= self.total_samples_current:
            self.current_year_idx = (self.current_year_idx + 1) % len(self.year_datasets)
            self.current_batch_idx = 0
            self._update_current_year_data()
            print(f"  → Switched to year {self.year_datasets[self.current_year_idx]['year']}")
        
        return X, y, A
    
    def _get_batch_geographic(self):
        """Get batch in geographic mode with state progression."""
        start_idx = self.current_batch_idx
        end_idx = min(start_idx + self.batch_size, self.total_samples_current)
        
        X = self.X_current[start_idx:end_idx]
        y = self.y_current[start_idx:end_idx]
        A = self.A_current[start_idx:end_idx]
        
        self.current_batch_idx = end_idx
        
        # If finished current state, move to next
        if self.current_batch_idx >= self.total_samples_current:
            self.current_state_idx = (self.current_state_idx + 1) % len(self.state_datasets)
            self.current_batch_idx = 0
            self._update_current_state_data()
            print(f"  → Switched to state {self.state_datasets[self.current_state_idx]['state']}")
        
        return X, y, A
    
    def _get_batch_static(self):
        """Get batch in static mode (no drift)."""
        if self.current_idx >= self.total_samples:
            # Reset and shuffle for new epoch
            self.current_idx = 0
            perm = np.random.permutation(self.total_samples)
            self.X_all = self.X_all[perm]
            self.y_all = self.y_all[perm]
            self.A_all = self.A_all[perm]
        
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, self.total_samples)
        
        X = self.X_all[start_idx:end_idx]
        y = self.y_all[start_idx:end_idx]
        A = self.A_all[start_idx:end_idx]
        
        self.current_idx = end_idx
        
        return X, y, A


def generate_folktables_batch(stream, drift_strength=0.0):
    """
    Wrapper function for compatibility with existing code.
    
    Args:
        stream: FolktablesDataStream object
        drift_strength: ignored (kept for compatibility)
    
    Returns:
        X, y, A: features, labels, sensitive attributes
    """
    return stream.get_batch()
