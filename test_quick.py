"""
Quick test script to verify test_layer_evolution.py works with a small number of images.
Run this before running the full 1000-image analysis.
"""

import sys
import os

# Modify the configuration in test_layer_evolution
original_file = "test_layer_evolution.py"
backup_file = "test_layer_evolution_backup.py"

# Read the original file
with open(original_file, 'r') as f:
    content = f.read()

# Create backup
with open(backup_file, 'w') as f:
    f.write(content)

# Modify NUM_IMAGES to 5 for testing
modified_content = content.replace('NUM_IMAGES = 1000', 'NUM_IMAGES = 5')
modified_content = modified_content.replace("OUTPUT_FILE = \"layer_evolution_results.json\"", 
                                           "OUTPUT_FILE = \"test_results.json\"")

# Write modified version
with open(original_file, 'w') as f:
    f.write(modified_content)

print("="*70)
print("RUNNING TEST WITH 5 IMAGES")
print("="*70)

# Run the test
exit_code = os.system("python test_layer_evolution.py")

# Restore original file
with open(backup_file, 'r') as f:
    original_content = f.read()

with open(original_file, 'w') as f:
    f.write(original_content)

# Clean up backup
os.remove(backup_file)

if exit_code == 0:
    print("\n" + "="*70)
    print("✅ TEST PASSED! The script works correctly.")
    print("="*70)
    print("\nYou can now run the full analysis with:")
    print("  python test_layer_evolution.py")
    print("\nTest results saved to: test_results.json")
else:
    print("\n" + "="*70)
    print("❌ TEST FAILED! Check the error messages above.")
    print("="*70)
    sys.exit(1)

