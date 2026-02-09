"""
Quick test script to verify Folktables integration.
This downloads a small sample and verifies everything works.
"""

import numpy as np
from src.data import FolktablesDataStream, generate_batch

def test_synthetic_data():
    """Test original synthetic data generation."""
    print("=" * 60)
    print("TEST 1: Synthetic Data")
    print("=" * 60)
    
    X, y, A = generate_batch(n=100, drift_strength=0.0)
    
    print(f"✓ Generated synthetic batch")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - A shape: {A.shape}")
    print(f"  - Group balance: A=0: {np.sum(A==0)}, A=1: {np.sum(A==1)}")
    print(f"  - Label balance: y=0: {np.sum(y==0)}, y=1: {np.sum(y==1)}")
    print()

def test_folktables_data():
    """Test Folktables data loading."""
    print("=" * 60)
    print("TEST 2: Folktables Data (California, Income, SEX)")
    print("=" * 60)
    print("Note: First run will download ~200MB of data...")
    print()
    
    try:
        # Initialize stream with small sample
        stream = FolktablesDataStream(
            task='income',
            states=['CA'],
            years=[2018],
            sensitive_attribute='SEX',
            batch_size=100
        )
        
        print(f"✓ Initialized Folktables stream")
        print(f"  - Total samples available: {stream.total_samples}")
        print(f"  - Feature dimensions: {stream.X_all.shape[1]}")
        print()
        
        # Get a batch
        X, y, A = stream.get_batch()
        
        print(f"✓ Retrieved batch")
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        print(f"  - A shape: {A.shape}")
        print(f"  - Group balance: A=0 (Female): {np.sum(A==0)}, A=1 (Male): {np.sum(A==1)}")
        print(f"  - Label balance: y=0 (<=50K): {np.sum(y==0)}, y=1 (>50K): {np.sum(y==1)}")
        print(f"  - Feature stats: mean={X.mean():.3f}, std={X.std():.3f}")
        print()
        
        # Test multiple batches
        X2, y2, A2 = stream.get_batch()
        print(f"✓ Retrieved second batch (sequential streaming works)")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection (first run downloads data)")
        print("2. Verify folktables is installed: pip3 install folktables")
        print("3. Check disk space (~200MB needed)")
        return False

def test_compatibility():
    """Test backward compatibility with existing code."""
    print("=" * 60)
    print("TEST 3: Backward Compatibility")
    print("=" * 60)
    
    # Test that old code still works
    from src.model import OnlineModel
    from src.fairness import demographic_parity
    
    X, y, A = generate_batch(n=200)
    
    model = OnlineModel()
    model.fit(X, y)
    y_pred = model.predict(X)
    dp = demographic_parity(y_pred, A)
    
    print(f"✓ OnlineModel works")
    print(f"✓ demographic_parity works")
    print(f"  - DP gap: {dp:.4f}")
    print()

if __name__ == "__main__":
    np.random.seed(42)
    
    print("\n" + "=" * 60)
    print("FOLKTABLES INTEGRATION TEST SUITE")
    print("=" * 60)
    print()
    
    # Test 1: Synthetic data (should always work)
    test_synthetic_data()
    
    # Test 2: Backward compatibility
    test_compatibility()
    
    # Test 3: Folktables (requires download)
    success = test_folktables_data()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if success:
        print("✓ All tests passed!")
        print("\nYou can now run:")
        print("  python run.py              # Synthetic data experiments")
        print("  python run_folktables.py   # Real data experiments")
    else:
        print("✓ Synthetic data works")
        print("✗ Folktables needs troubleshooting (see above)")
        print("\nYou can still run synthetic experiments:")
        print("  python run.py")
    print()
