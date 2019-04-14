import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ground_truths import ground_truths_politicians
import warnings

warnings.filterwarnings('ignore')

# author_handle : (social %, economic %, error)
def plotrunNolans(df, testuser):
    df['Type'] = 'Model Estimation'

    df2 = pd.DataFrame.from_dict(ground_truths_politicians, orient='index')
    df2 = df2.reset_index()
    df2.columns = ['author_handle','s_score','e_score']
    df2['Type'] = 'Politician'

    df1 = df.append(df2, ignore_index=True)

    lines = [
        (-.1, .2, .5, .5),
        (.8, 1.1, .5, .5),
        (.5, .5, 1.1, .8),
        (.5, .5, -.1, .2),
        (.2, .5, .5, .2),
        (.5, .2, .8, .5),
        (.8, .5, .5, .8),
        (.5, .8, .2, .5)
    ]
    for x1, y1, x2, y2 in lines:
        plt.plot([x1, y1], [x2, y2], 'k-', color='black')

    ax = sns.scatterplot(x='e_score',
                         y='s_score',
                         style='Type',
                         hue='Type',
                         data=df1,
                         s=200)

    for idx in range(0, df1.shape[0]):
        if df1.author_handle[idx] == testuser:
            if df1.Type[idx] == 'Model Estimation':
                loc = [df1.e_score[idx] + 0.002, df1.s_score[idx] +.005]
            else:
                loc = [df1.e_score[idx] + 0.002, df1.s_score[idx] -.03]

            ax.text(loc[0],
                    loc[1],
                    df1.author_handle[idx],
                    horizontalalignment='left',
                    fontsize=12,
                    color='darkslategrey')

    ax.set(ylabel='Ground Truth Social %', xlabel='Ground Truth Economic %')
    ax.set_title('Ground Truth Political Affiliations Estimation via Linear Regression over By-User Feature Matrix')
    ax.set_aspect(1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()