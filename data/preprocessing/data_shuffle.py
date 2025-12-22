#!/usr/bin/env python3
"""
Data Shuffling Script for Qwen-MoE
모든 도메인의 데이터를 섞어서 통합된 train/test 파일 생성
"""

import json
import os
import random
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✅ Loaded {len(data)} samples from {file_path}")
        return data
    except Exception as e:
        logger.error(f"❌ Failed to load {file_path}: {e}")
        return []

def save_json_data(data: List[Dict[str, Any]], file_path: str):
    """Save JSON data to file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved {len(data)} samples to {file_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save {file_path}: {e}")

def add_domain_label(data: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
    """Add domain label to each sample"""
    for sample in data:
        sample['domain'] = domain
    return data

def shuffle_and_merge_data():
    """Shuffle and merge all domain data"""
    logger.info("🔄 Starting data shuffling and merging process...")
    
    # Define data paths
    data_config = {
        'code': {
            'train': 'processed/code/code_train.json',
            'test': 'processed/code/code_test.json'
        },  
        'medical': {
            'train': 'processed/medical/medical_train.json',
            'test': 'processed/medical/medical_test.json'
        },
        'math': {
            'train': 'processed/math/math_train.json',
            'test': 'processed/math/math_test.json'
        },
        'law': {
            'train': 'processed/law/law_train.json',
            'test': 'processed/law/law_test.json'
        }
    }
    
    # Load all training data
    logger.info("📂 Loading training data...")
    all_train_data = []
    train_stats = {}
    
    for domain, paths in data_config.items():
        train_data = load_json_data(paths['train'])
        if train_data:
            # Add domain label
            train_data = add_domain_label(train_data, domain)
            all_train_data.extend(train_data)
            train_stats[domain] = len(train_data)
            logger.info(f"  {domain}: {len(train_data):,} samples")
    
    # Load all test data
    logger.info("📂 Loading test data...")
    all_test_data = []
    test_stats = {}
    
    for domain, paths in data_config.items():
        test_data = load_json_data(paths['test'])
        if test_data:
            # Add domain label
            test_data = add_domain_label(test_data, domain)
            all_test_data.extend(test_data)
            test_stats[domain] = len(test_data)
            logger.info(f"  {domain}: {len(test_data):,} samples")
    
    # Print statistics
    logger.info("📊 Data Statistics:")
    logger.info("  Training Data:")
    for domain, count in train_stats.items():
        logger.info(f"    {domain}: {count:,} samples")
    logger.info(f"    Total: {sum(train_stats.values()):,} samples")
    
    logger.info("  Test Data:")
    for domain, count in test_stats.items():
        logger.info(f"    {domain}: {count:,} samples")
    logger.info(f"    Total: {sum(test_stats.values()):,} samples")
    
    # Shuffle data
    logger.info("🔀 Shuffling data...")
    random.seed(42)  # For reproducibility
    random.shuffle(all_train_data)
    random.shuffle(all_test_data)
    
    # Save shuffled data
    logger.info("💾 Saving shuffled data...")
    
    # Save training data
    train_output_path = 'data/total/total_train.json'
    save_json_data(all_train_data, train_output_path)
    
    # Save test data
    test_output_path = 'data/total/total_test.json'
    save_json_data(all_test_data, test_output_path)
    
    # Verify saved data
    logger.info("🔍 Verifying saved data...")
    verify_train = load_json_data(train_output_path)
    verify_test = load_json_data(test_output_path)
    
    # Count domains in shuffled data
    train_domain_counts = {}
    test_domain_counts = {}
    
    for sample in verify_train:
        domain = sample.get('domain', 'unknown')
        train_domain_counts[domain] = train_domain_counts.get(domain, 0) + 1
    
    for sample in verify_test:
        domain = sample.get('domain', 'unknown')
        test_domain_counts[domain] = test_domain_counts.get(domain, 0) + 1
    
    logger.info("✅ Final Statistics:")
    logger.info("  Shuffled Training Data:")
    for domain, count in train_domain_counts.items():
        logger.info(f"    {domain}: {count:,} samples")
    logger.info(f"    Total: {len(verify_train):,} samples")
    
    logger.info("  Shuffled Test Data:")
    for domain, count in test_domain_counts.items():
        logger.info(f"    {domain}: {count:,} samples")
    logger.info(f"    Total: {len(verify_test):,} samples")
    
    logger.info("🎉 Data shuffling and merging completed successfully!")
    
    return {
        'train_path': train_output_path,
        'test_path': test_output_path,
        'train_count': len(verify_train),
        'test_count': len(verify_test),
        'train_domain_counts': train_domain_counts,
        'test_domain_counts': test_domain_counts
    }

def main():
    """Main function"""
    logger.info("🚀 Starting Qwen-MoE Data Shuffling Process")
    
    try:
        result = shuffle_and_merge_data()
        
        logger.info("📋 Summary:")
        logger.info(f"  Training file: {result['train_path']}")
        logger.info(f"  Test file: {result['test_path']}")
        logger.info(f"  Total training samples: {result['train_count']:,}")
        logger.info(f"  Total test samples: {result['test_count']:,}")
        
        logger.info("✅ Process completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Process failed: {e}")
        raise

if __name__ == "__main__":
    main()
