import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ground_truths import ground_truths_politicians
import warnings
import sqlite3
import matplotlib
matplotlib.rcParams['figure.figsize'] = [8, 5]

warnings.filterwarnings('ignore')

# author_handle : (social %, economic %, error)
def plotrunNolans(df, testuser):
    df['Type'] = 'Model Estimation'

    df2 = pd.DataFrame.from_dict({testuser: ground_truths_politicians[testuser]}, orient='index')
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

    ax.set(ylabel='Ground Truth Social %', xlabel='Ground Truth Economic %')
    ax.set_title('{}\'s Political Affiliation Estimation via Linear Regression over By-User Feature Matrix'.format(
        testuser))
    plt.ylim(-.1, 1.1)
    plt.xlim(-.1, 1.1)
    ax.grid(True)
    ax.set_aspect(1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.figure(figsize=(8.0,6.0))
    #plt.savefig(r"viz\{}.png".format(testuser), dpi=100)
    plt.show()

def datasetviz():
    conn = sqlite3.connect('tweet_data.db')
    df = pd.read_sql('''select user_id, author_handle, e_score, s_score, real_party, score_freq from user;''',
                     conn,
                     index_col='user_id')
    df['Nolan Chart Classification'] = df['real_party']
    df['Number of Users at a Given Point'] = df['score_freq']
    df = df[df['score_freq'] > 0]
    df['Number of Users at a Given Point'] += 2

    lines = [
        (-.1,.2,.5,.5),
        (.8,1.1,.5,.5),
        (.5,.5,1.1,.8),
        (.5,.5,-.1,.2),
        (.2,.5,.5,.2),
        (.5,.2, .8,.5),
        (.8,.5,.5,.8),
        (.5,.8,.2,.5)
    ]

    for x1, y1, x2, y2 in lines:
        plt.plot([x1, y1], [x2, y2], 'k-', color='black')
        #ax.plot(x1, y1, x2, y2, marker='o')

    ax = sns.scatterplot(x="e_score",
                         y='s_score',
                         hue='Nolan Chart Classification',
                         hue_order=['Left Liberal','Centrist','Libertarian','Right Conservative','Populist'],
                         #size='''Number of Users at a Given Point''',
                         #sizes=(100,650),
                         data=df)

    # locs = [
    #         [.25,.1,'Liberal'],
    #         [],
    #         [],
    #         []
    #         ]
    #
    # for loc in locs:
    #     ax.text(loc[0],
    #             loc[1],
    #             loc[3],
    #             horizontalalignment='center',
    #             fontsize=11,
    #             color='darkslategrey')

    plt.ylabel('Ground Truth Social %')
    plt.xlabel('Ground Truth Economic %')
    plt.title('Nolan Chart Position Clustering of Dataset Politicians and Pundits')
    plt.ylim(-.1, 1.1)
    plt.xlim(-.1, 1.1)
    ax.grid(True)
    ax.set_aspect(1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax.get_legend().set_visible(False)
    plt.show()

def plotrunNolans2(df):
    # df2 = pd.DataFrame.from_dict(ground_truths_politicians, orient='index')
    # df2 = df2.reset_index()
    # df2.columns = ['author_handle','s_score','e_score']
    # df2['Type'] = 'Training User (Politician)'

    # df = df.append(df2, ignore_index=True)

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
                         style='Failed',
                         hue='Failed',
                         data=df,
                         s=120)

    already_written = []
    df = df[df['Failed'] == True]
    for line in range(0, df.shape[0]):
        if df.author_handle[line] not in ground_truths_politicians:
            loc = [df.e_score[line] + 0.002, df.s_score[line] + .005]
            if loc not in already_written:
                ax.text(loc[0],
                        loc[1],
                        df.author_handle[line],
                        horizontalalignment='left',
                        fontsize=10,
                        color='darkslategrey')
                already_written.append(loc)
            else:
                while loc in already_written:
                    loc = [loc[0], loc[1] + .02]
                ax.text(loc[0],
                        loc[1],
                        df.author_handle[line],
                        horizontalalignment='left',
                        fontsize=10,
                        color='darkslategrey')
                already_written.append(loc)

    ax.set(ylabel='Ground Truth Social %', xlabel='Ground Truth Economic %')
    ax.set_title('Pundits\' Model Estimations via Linear Regression + PCA')
    plt.ylim(-.1, 1.1)
    plt.xlim(-.1, 1.1)
    ax.grid(True)
    ax.set_aspect(1)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.figure(figsize=(8.0,6.0))
    #plt.savefig(r"viz\{}.png".format(testuser), dpi=100)
    plt.show()