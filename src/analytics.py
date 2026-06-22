import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

#                   Data summaries

def category_distribution(df):
    counts = df['category'].value_counts()
    pct = df['category'].value_counts(normalize=True) * 100
    return pd.DataFrame({'count': counts, 'percentage': pct.round(2)})

def most_common_issue_types(df, top_n=3):
    return df['category'].value_counts().head(top_n)

def priority_breakdown(df):
    counts = df['priority'].value_counts()
    pct = df['priority'].value_counts(normalize=True) * 100
    return pd.DataFrame({'count': counts, 'percentage': pct.round(2)})


#                   Plots

def plot_category_distribution(df):
    counts = df['category'].value_counts()
    fig_ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind='bar', ax=fig_ax[1])
    fig_ax[1].set_title('Ticket Category Distribution')
    fig_ax[1].set_xlabel('Category')
    fig_ax[1].set_ylabel('Number of Tickets')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

def plot_priority_breakdown(df):
    priority_order = ['Low', 'Medium', 'High', 'Critical']
    counts = df['priority'].value_counts().reindex(priority_order)
    fig_ax = plt.subplots(figsize=(6, 5))
    counts.plot(kind='bar', ax=fig_ax[1], color=['#55A868', '#DD8452', '#C44E52', '#8C0000'])
    fig_ax[1].set_title('Priority Level Breakdown')
    fig_ax[1].set_xlabel('Priority')
    fig_ax[1].set_ylabel('Number of Tickets')
    plt.xticks(rotation=0)
    plt.tight_layout()

def plot_most_common_issue_types(df, top_n=3):
    counts = df['category'].value_counts().head(top_n).sort_values()
    fig_ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind='barh', ax=fig_ax[1])
    fig_ax[1].set_title(f'Top {top_n} Most Common Issue Types')
    fig_ax[1].set_xlabel('Number of Tickets')
    fig_ax[1].set_ylabel('Category')
    plt.tight_layout()


if __name__=='__main__':
    curr_dir = Path(__file__).parent
    df = pd.read_csv(curr_dir.parent/'data/support_tickets_dataset_v2.csv')

    print('Category Distribution:')
    print(category_distribution(df))
    print('\nPriority Breakdown:')
    print(priority_breakdown(df))
    print('\nMost Common Issue Types:')
    print(most_common_issue_types(df))

    plot_category_distribution(df)
    plot_priority_breakdown(df)
    plot_most_common_issue_types(df)
    plt.show()