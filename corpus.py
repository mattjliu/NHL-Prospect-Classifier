import pandas as pd
import os
import re
import io
pd.options.mode.chained_assignment = None


def create_corpus(verbose=True, threshold='mean'):

    reports = pd.read_csv('data/reports/reports_v2.csv', encoding='utf-8')
    stats = pd.read_csv('data/stats/stats_v2.csv', encoding='utf-8')

    # Format names for fileids
    reports['name'] = reports['name'].str.replace(' ', '_')
    stats['name'] = stats['name'].str.replace(' ', '_')
    reports['name'] = reports['name'].str.replace('"', '')
    stats['name'] = stats['name'].str.replace('"', '')

    # Shift draft numbers because of new jersey's forfeited pick
    reports_shift = reports[(reports.draft_year == 2011) & (reports.draft_num >= 69)]
    reports_shift['draft_num'] += 1
    reports = reports[~reports['name'].isin(reports_shift['name'])]
    reports = pd.concat([reports, reports_shift])

    reports2019 = reports[reports.draft_year == 2019]
    reports_hist = reports[reports.draft_year != 2019]
    merged = pd.merge(reports_hist.drop(columns='name'), stats, on=['draft_year', 'draft_num'], how='inner')

    # Define threshold games played depending on agg function or constant
    if isinstance(threshold, str):
        grouped_stat = merged.groupby('draft_year')['GP'].agg(threshold)
        threshold_gp = merged.apply(lambda x: grouped_stat.loc[x.draft_year, ], axis=1)
        if verbose:
            print(f'================== {threshold.capitalize()} Games Played by Draft Year ==================')
            print(grouped_stat)
    else:
        threshold_gp = threshold
    merged['NHL'] = merged['GP'] > threshold_gp

    # Define train and valid set
    mask = (merged['draft_year'] >= 2016) & (merged['NHL'] == False)
    reports2019['draft_team'] = 'TBD'
    valid = pd.concat([reports2019, merged[mask][['draft_num', 'draft_year', 'draft_team', 'name', 'report']]], sort=False)
    train = merged[~mask]

    if not os.path.exists('data/merged'):
        os.makedirs('data/merged')
    train.to_csv('data/merged/train.csv')
    valid.to_csv('data/merged/valid.csv')
    if verbose:
        print('Merged data created successfully')

    if not os.path.exists('data/NHLcorpus/true'):
        os.makedirs('data/NHLcorpus/true')
    if not os.path.exists('data/NHLcorpus/false'):
        os.makedirs('data/NHLcorpus/false')
    try:
        for _, row in train.iterrows():
            with io.open(os.path.join('data/NHLcorpus', str(row.NHL), '{}_{}.txt'.format(row['name'], row.draft_year)), 'w', encoding='utf-8') as f:
                f.write(row.report)
                f.close()
    except OSError as e:
        print(e)
        return

    if verbose:
        print('Corpus created successfully')


if __name__ == '__main__':
    create_corpus()
