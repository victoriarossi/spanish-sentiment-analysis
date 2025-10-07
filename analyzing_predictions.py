import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

# Find all CSV files in the current directory
csv_files = glob.glob('./predictions/pretrained_in_ES/*_predictions.csv')

print("Files found for analysis:")
print(csv_files)

csv_original_files = ["./datasets/full datasets/reviews_filmaffinity_with_annotations.csv",
                      "./datasets/full datasets/sentiment_analysis_dataset_with_annotations.csv",
                      "./datasets/full datasets/tweets_with_annotations.csv",
                      "./datasets/full datasets/train_textos_turisticos.csv"]

results = {}

# Define sentiment order and mapping
sentiment_order = ['VERY NEGATIVE', 'NEGATIVE', 'NEUTRAL', 'POSITIVE', 'VERY POSITIVE']
sentiment_to_num = {label: i for i, label in enumerate(sentiment_order)}

all_data = []

i = 0
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Calculate accuracy (where pred_label equals gold_label)
    accuracy = (df['pred_label'] == df['gold_label']).mean() * 100
    
    # Store result
    results[file] = accuracy
    
    print(f"{file}: {accuracy:.2f}% accuracy, size of file: {len(pd.read_csv(csv_original_files[i]))} rows")
    
    # Add file identifier for combined analysis
    df['file'] = file
    all_data.append(df)
    i = i + 1

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)

# Create visualization 1: Bar chart
plt.figure(figsize=(10, 6))
names = ['Reviews FilmAffinity', 'Sentiment Analysis Dataset', 'Tweets with Annotations', 'Train Textos Turisticos']
plt.bar(names, results.values(), color=['blue', 'red','green'])
plt.xlabel('Files')
plt.ylabel('Accuracy (%)')
plt.title('BERT Sentiment Prediction Accuracy by File')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.tight_layout()

# Add percentage labels on bars
for file, acc in results.items():
    plt.text(list(results.keys()).index(file), acc + 1, f'{acc:.1f}%', 
             ha='center', va='bottom')

plt.grid(axis='y', alpha=0.3)
plt.savefig('./analysis/accuracy_results.png', dpi=300, bbox_inches='tight')
# plt.show()

idx = 0
# Create confusion matrices for each file
for file in csv_files:
    df = pd.read_csv(file)
    
    # Create confusion matrix for this file
    plt.figure(figsize=(10, 8))
    df['gold_label'] = df['gold_label'].str.upper()
    df['pred_label'] = df['pred_label'].str.upper()
    confusion_matrix = pd.crosstab(df['gold_label'], 
                                    df['pred_label'], 
                                    rownames=['Actual'], 
                                    colnames=['Predicted'])
    
    # Reorder to sentiment scale
    confusion_matrix = confusion_matrix.reindex(index=sentiment_order, 
                                               columns=sentiment_order, 
                                               fill_value=0)
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix: {names[idx]}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./analysis/confusion_matrix_{names[idx]}.png', dpi=300, bbox_inches='tight')
    
    # Create error distribution heatmap for this file
    plt.figure(figsize=(10, 8))
    error_matrix = confusion_matrix.copy().astype(float)
    for i, actual in enumerate(sentiment_order):
        row_sum = error_matrix.loc[actual].sum()
        if row_sum > 0:
            error_matrix.loc[actual] = error_matrix.loc[actual] / row_sum * 100
    
    sns.heatmap(error_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
    
    print(f"Tried to save at i= {idx}")
    plt.title(f'Prediction Distribution: {names[idx]} (% within each row)')
    plt.tight_layout()
    plt.savefig(f'./analysis/error_distribution_{names[idx]}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Calculate statistics for this file
    df['gold_num'] = df['gold_label'].map(sentiment_to_num)
    df['pred_num'] = df['pred_label'].map(sentiment_to_num)
    df['distance'] = abs(df['gold_num'] - df['pred_num'])
    
    print(f"\n--- Statistics for {file} ---")
    print(f"Exact match accuracy: {(df['distance'] == 0).mean() * 100:.2f}%")
    print(f"Off by 1 level: {(df['distance'] == 1).mean() * 100:.2f}%")
    print(f"Off by 2 levels: {(df['distance'] == 2).mean() * 100:.2f}%")
    print(f"Off by 3 levels: {(df['distance'] == 3).mean() * 100:.2f}%")
    print(f"Off by 4 levels: {(df['distance'] == 4).mean() * 100:.2f}%")
    
    # Calculate weighted accuracy
    weights = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25, 4: 0.0}
    df['weighted_score'] = df['distance'].map(weights)
    weighted_accuracy = df['weighted_score'].mean() * 100
    print(f"Weighted accuracy (partial credit): {weighted_accuracy:.2f}%")
    idx = idx + 1

plt.show()

# Print overall statistics
print(f"\n=== OVERALL STATISTICS ===")
print(f"Overall accuracy across all files: {sum(results.values())/len(results):.2f}%")

combined_df['gold_num'] = combined_df['gold_label'].map(sentiment_to_num)
combined_df['pred_num'] = combined_df['pred_label'].map(sentiment_to_num)
combined_df['distance'] = abs(combined_df['gold_num'] - combined_df['pred_num'])

print(f"\nCombined - Exact match accuracy: {(combined_df['distance'] == 0).mean() * 100:.2f}%")
print(f"Combined - Off by 1 level: {(combined_df['distance'] == 1).mean() * 100:.2f}%")
print(f"Combined - Off by 2 levels: {(combined_df['distance'] == 2).mean() * 100:.2f}%")
print(f"Combined - Off by 3 levels: {(combined_df['distance'] == 3).mean() * 100:.2f}%")
print(f"Combined - Off by 4 levels: {(combined_df['distance'] == 4).mean() * 100:.2f}%")

weights = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25, 4: 0.0}
combined_df['weighted_score'] = combined_df['distance'].map(weights)
weighted_accuracy = combined_df['weighted_score'].mean() * 100
print(f"Combined - Weighted accuracy (partial credit): {weighted_accuracy:.2f}%")