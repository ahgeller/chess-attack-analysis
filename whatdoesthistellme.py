import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load and combine datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
csv1_path = os.path.join(script_dir, "CSV files", "TESTatcData.csv")
try:
    print("Finding TESTatcDataLichess")
    csv2_path = os.path.join(script_dir, "CSV files", "TESTatcDataLichess.csv")
    print("Success! Now loading data.")

    df2 = pd.read_csv(csv2_path)
    print("Using TESTatcDataLichess")
except:
    print("Only using TESTatcDATA")
    df2 = pd.DataFrame()

 

df1 = pd.read_csv(csv1_path)
# df2.to_parquet('large-file.parquet', compression='gzip')

# Standardize columns before combining
# For df1: if it has white_rating and black_rating, calculate average_rating
if 'white_rating' in df1.columns and 'black_rating' in df1.columns:
    df1['average_rating'] = (df1['white_rating'] + df1['black_rating']) / 2

# For df2: if it has average_rating but not individual ratings, add NaN columns
if 'average_rating' in df2.columns and 'white_rating' not in df2.columns:
    df2['white_rating'] = np.nan
    df2['black_rating'] = np.nan

# Add missing columns if they don't exist in either
if 'opening_name' not in df1.columns:
    df1['opening_name'] = np.nan
if 'opening_name' not in df2.columns:
    df2['opening_name'] = np.nan
if 'result' not in df1.columns:
    df1['result'] = np.nan
if 'result' not in df2.columns:
    df2['result'] = np.nan

# Combine datasets
df = pd.concat([df1, df2], ignore_index=True)

# Aggregate to game level
agg_dict = {
    'white_attack_value': 'mean',
    'black_attack_value': 'mean',
    'attack_differential': 'mean',
    'average_rating': 'first'
}

# Add optional columns if they exist
if 'white_rating' in df.columns:
    agg_dict['white_rating'] = 'first'
if 'black_rating' in df.columns:
    agg_dict['black_rating'] = 'first'
if 'opening_name' in df.columns:
    agg_dict['opening_name'] = 'first'
if 'result' in df.columns:
    agg_dict['result'] = 'first'

game_stats = df.groupby('game_id').agg(agg_dict).reset_index()

# Rename columns
game_stats = game_stats.rename(columns={
    'white_attack_value': 'white_avg',
    'black_attack_value': 'black_avg',
    'attack_differential': 'diff_avg'
})

# 1. Attack Progression Over Moves
moves_avg = df.groupby('move_number')[['white_attack_value', 'black_attack_value']].mean()
moves_avg = moves_avg[moves_avg.index <= 250]

plt.figure(figsize=(12, 5))
plt.plot(moves_avg.index, moves_avg['white_attack_value'], label='White', color='blue', linewidth=2)
plt.plot(moves_avg.index, moves_avg['black_attack_value'], label='Black', color='red', linewidth=2)
plt.xlabel('Move Number')
plt.ylabel('Average Attack Value')
plt.title('Attack Values Throughout Game')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('attack_progression.png', dpi=150)
plt.show()

# 2. Opening Analysis
if 'opening_name' in game_stats.columns and game_stats['opening_name'].notna().any():
    opening_data = game_stats[game_stats['opening_name'].notna()]
    if len(opening_data) > 0:
        top_openings = opening_data['opening_name'].value_counts().head(15).index
        opening_stats = opening_data[opening_data['opening_name'].isin(top_openings)].groupby('opening_name').agg(
            differential=('diff_avg', 'mean'),
            count=('game_id', 'count')
        ).sort_values('differential')
        
        plt.figure(figsize=(10, 7))
        colors = ['red' if x < 0 else 'blue' for x in opening_stats['differential']]
        plt.barh(range(len(opening_stats)), opening_stats['differential'], color=colors, alpha=0.7)
        plt.axvline(0, color='black', linewidth=1.5)
        plt.yticks(range(len(opening_stats)), 
                   [f"{name} (n={opening_stats.loc[name, 'count']})" for name in opening_stats.index])
        plt.xlabel('Average Attack Differential (White - Black)')
        plt.title('Which Openings Give Attack Advantage?')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('opening_attack_advantage.png', dpi=150)
        plt.show()

# 3. Attack vs Outcome
if 'result' in game_stats.columns and game_stats['result'].notna().any():
    result_stats = game_stats[game_stats['result'].isin(['white', 'black', 'draw'])].groupby('result').agg({
        'white_avg': 'mean',
        'black_avg': 'mean',
        'diff_avg': 'mean'
    })
    
    if len(result_stats) > 0:
        result_order = ['white', 'draw', 'black']
        result_stats = result_stats.reindex([r for r in result_order if r in result_stats.index])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(result_stats))
        
        axes[0].bar(x - 0.175, result_stats['white_avg'], 0.35, label='White', color='blue', alpha=0.7)
        axes[0].bar(x + 0.175, result_stats['black_avg'], 0.35, label='Black', color='red', alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([r.capitalize() for r in result_stats.index])
        axes[0].set_ylabel('Average Attack Value')
        axes[0].set_title('Attack Values by Game Outcome')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        colors = ['blue' if r == 'white' else 'gray' if r == 'draw' else 'red' for r in result_stats.index]
        axes[1].bar(x, result_stats['diff_avg'], color=colors, alpha=0.7)
        axes[1].axhline(0, color='black', linewidth=1.5, linestyle='--')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([r.capitalize() for r in result_stats.index])
        axes[1].set_ylabel('Attack Differential')
        axes[1].set_title('Attack Advantage by Outcome')
        axes[1].grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('attack_vs_outcome.png', dpi=150)
        plt.show()

# 4. Rating vs Attack
if 'average_rating' in game_stats.columns:
    rating_data = game_stats[game_stats['average_rating'].notna()]
    if len(rating_data) > 0:
        rating_data = rating_data[(rating_data['average_rating'] >= 800) & (rating_data['average_rating'] <= 2800)]
        rating_data['rating_cat'] = pd.cut(rating_data['average_rating'], 
                                           bins=[0, 1200, 1600, 2000, 2400, 3000],
                                           labels=['<1200', '1200-1600', '1600-2000', '2000-2400', '2400+'])
        
        rating_data['total_attack'] = rating_data['white_avg'] + rating_data['black_avg']
        rating_stats = rating_data.groupby('rating_cat', observed=True).agg({
            'white_avg': 'mean',
            'black_avg': 'mean',
            'total_attack': 'mean'
        })
        
        if len(rating_stats) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            x = np.arange(len(rating_stats))
            
            axes[0].bar(x - 0.175, rating_stats['white_avg'], 0.35, label='White', color='blue', alpha=0.7)
            axes[0].bar(x + 0.175, rating_stats['black_avg'], 0.35, label='Black', color='red', alpha=0.7)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(rating_stats.index)
            axes[0].set_ylabel('Average Attack Value')
            axes[0].set_title('Attack Levels by Player Skill')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
            
            axes[1].plot(x, rating_stats['total_attack'], marker='o', color='purple', linewidth=2.5, markersize=10)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(rating_stats.index)
            axes[1].set_ylabel('Total Attack (White + Black)')
            axes[1].set_title('Combined Attack by Skill Level')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('rating_vs_attack.png', dpi=150)
            plt.show()

# 5. White vs Black Correlation
plt.figure(figsize=(8, 8))
sample = game_stats#.sample(min(100000, len(game_stats)))
plt.hexbin(sample['white_avg'], sample['black_avg'], gridsize=50, cmap='YlOrRd', mincnt=1, bins='log')
plt.plot([0, 20], [0, 20], 'b--', linewidth=2, label='Equal attack', alpha=0.7)
corr = game_stats[['white_avg', 'black_avg']].corr().iloc[0, 1]
plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.xlabel('White Average Attack')
plt.ylabel('Black Average Attack')
plt.title('White vs Black Attack Correlation')
plt.colorbar(label='Number of Games (log scale)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.tight_layout()
plt.savefig('white_vs_black_correlation.png', dpi=150)
plt.show()

# 6. Attack Distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(game_stats['white_avg'], bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(game_stats['white_avg'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {game_stats["white_avg"].median():.2f}')
axes[0].set_xlabel('Average White Attack')
axes[0].set_ylabel('Number of Games')
axes[0].set_title('White Attack Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(game_stats['black_avg'], bins=50, color='red', alpha=0.7, edgecolor='black')
axes[1].axvline(game_stats['black_avg'].median(), color='blue', linestyle='--', linewidth=2,
                label=f'Median: {game_stats["black_avg"].median():.2f}')
axes[1].set_xlabel('Average Black Attack')
axes[1].set_ylabel('Number of Games')
axes[1].set_title('Black Attack Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].hist(game_stats['diff_avg'], bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[2].axvline(0, color='black', linestyle='--', linewidth=2)
axes[2].axvline(game_stats['diff_avg'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {game_stats["diff_avg"].median():.2f}')
axes[2].set_xlabel('Attack Differential (White - Black)')
axes[2].set_ylabel('Number of Games')
axes[2].set_title('Attack Advantage Distribution')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('attack_distributions.png', dpi=150)
plt.show()
