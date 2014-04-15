from datetime import date


class User(object):

    def __init__(self, uid, first_name, last_name):
        self.is_user_active = True
        self.uid = uid
        self.first_name = first_name
        self.last_name = last_name
        # 1 = woman, 2 = man
        self.sex = -1
        self.age = -1
        self.university = -1
        self.graduation = 0
        self.relation = -1
        self.status_len = -1
        self.groups = -1
        self.user_photos = -1
        self.user_videos = -1
        self.videos = -1
        self.notes = -1
        self.subscriptions = -1
        self.mutual_friends = -1
        self.audios = -1
        self.photos = -1
        self.followers = -1
        self.albums = -1
        self.friends = -1
        self.pages = -1
        self.likes = -1
        self.frequency = -1
        self.emotions = -1

    def set_age(self, birth_date):
        if birth_date:
            self.age = int((date.today() - birth_date).days / 365.2425)

    def set_frequency(self, post_date):
        if post_date:
            td1 = date.fromtimestamp(post_date)
            frequency = int((date.today() - td1).days)
            if frequency < 11:
                #hyperactive user
                self.frequency = 0
            elif frequency < 31:
                #active user
                self.frequency = 1
            elif frequency < 75:
                self.frequency = 2
            elif frequency < 180:
                self.frequency = 3
            elif frequency < 365:
                self.frequency = 4
            else:
                self.frequency = 5

    def to_tsv(self):
        return u'\t'.join([self.spaces(self.uid), self.spaces(self.last_name), unicode(self.sex),
                           unicode(self.age), unicode(self.likes), unicode(self.relation), unicode(self.groups)])

    @staticmethod
    def spaces(str):
        if isinstance(str, int):
            if str < 10000000:
                return "    " + unicode(str)
            else:
                return unicode(str)
        if len(str) < 8:
            return "    " + unicode(str)
        return unicode(str)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return u"User('{uid}', '{first}', '{last}')".format(uid=self.uid, first=self.first_name, last=self.last_name)

    def to_file(self):
        return [self.uid, self.sex, self.age, self.relation, self.status_len, self.emotions,
        self.videos, self.notes, self.subscriptions, self.mutual_friends, self.audios,
        self.photos, self.followers, self.albums, self.friends, self.pages, self.likes, self.graduation,
        self.frequency, self.user_photos, self.user_videos, self.groups]