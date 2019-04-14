def returnNonsense(author_handle, s_score, e_score):
    if e_score < 0 or s_score < 0 or e_score > 1 or s_score > 1:
        return True
    elif author_handle == 'RobDelaney':
        if s_score < .5 or e_score > .5:
            return True
        else:
            return ''
    elif author_handle == 'TomiLahren':
        if e_score < .5 or s_score > .5:
            return True
        else:
            return ''
    elif author_handle == 'RealSaavedra':
        if e_score < .5 or s_score > .5:
            return True
        else:
            return ''
    elif author_handle == 'KrangTNelson':
        if s_score < .5 or e_score > .5:
            return True
        else:
            return ''
    elif author_handle == 'JuliaIoffe':
        if e_score > .7 or s_score < .4:
            return True
        else:
            return ''
    elif author_handle == 'nprpolitics':
        if s_score < .3 or e_score > .7:
            return True
        else:
            return ''
    elif author_handle == 'reason':
        if e_score < .4 or s_score < .4:
            return True
        else:
            return ''
    elif author_handle == 'lizmair':
        if s_score < .3 or e_score < .4:
            return True
        else:
            return ''
    elif author_handle == 'KellyannePolls':
        if e_score < .5 or s_score > .5:
            return True
        else:
            return ''
    elif author_handle == 'MichelleMalkin':
        if s_score > .5 or e_score < .5:
            return True
        else:
            return ''
    elif author_handle == 'Heritage':
        if e_score < .4 or s_score > .6:
            return True
        else:
            return ''
    elif author_handle == 'RedState':
        if e_score < .5 or s_score > .5:
            return True
        else:
            return ''
    elif author_handle == 'GlennBeck':
        if s_score > .5 or e_score < .5:
            return True
        else:
            return ''
    elif author_handle == 'IngrahamAngle':
        if e_score < .5 or s_score > .5:
            return True
        else:
            return ''
    elif author_handle == 'JoeNBC':
        if e_score < .4 or s_score > .8:
            return True
        else:
            return ''
    elif author_handle == 'JonathanLKrohn':
        if e_score < .3 or s_score < .3:
            return True
        else:
            return ''
    elif author_handle == 'demsocialists':
        if s_score <.6 or e_score >.4:
            return True
        else:
            return ''
    elif author_handle == 'PostOpinions':
        if e_score >.8 or s_score <.3:
            return True
        else:
            return ''
    else:
        return ''
