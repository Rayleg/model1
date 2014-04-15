"""
VKontake HTTP API implementation stub
http://vk.com/pages?oid=-1&p=%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5_%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D0%BE%D0%B2_API
"""

__author__ = 'Nikolay Anokhin'

import datetime
import user
from api import Api
import time
import csv


class VkApi(Api):

    endpoint = "https://api.vk.com/method/{method}"

    def get_friend_ids(self, uid):
        json = self.call("friends.get", uid=uid)
        for friend_id in json.get("response", []):
            yield str(friend_id)

    def get_json_from_id(self, uid):
        json = self.call("users.get", uids=uid, fields="uid,first_name,last_name,sex,bdate,education,\
                                                        counters, relation, status")
        #print json
        for user_json in json.get("response", []):
            #returns first i suppose
            return user_json

    def get_all_users(self, uid_list):
        for uid in uid_list:
            print "Starting: " + str(uid)
            u = self.json_to_user(self.get_json_from_id(uid))
            time.sleep(0.2)
            if u.is_user_active:
                self.get_all_photo_comments(uid, u)
                time.sleep(0.2)
                self.get_wall_comments(uid, u)
                time.sleep(0.2)
            yield u

    def get_all_photo_comments(self, uid, u):
        # permission: photos
        json = self.call("photos.getAllComments", count=50, owner_id=uid)
        emotions = 0
        c = 0
        for item in json.get("response", []):
            #print item
            c += 1
            msg = item.get('message')
            for ch in msg:
                if ch in [')', '!', 'D']:
                    emotions += 1
        if c == 0:
            u.emotions = 0
        else:
            u.emotions = emotions*5/c

    def get_wall_comments(self, uid, u):
        json = self.call("wall.get", count=5, owner_id=uid, filter='owner')
        time.sleep(0.1)
        likes = 0
        date = 0
        c = 0
        for post in json.get("response", [])[1:]:
            c += 1
            lks = post.get("likes")
            likes += lks.get("count")
            date = post.get('date')
        u.set_frequency(date)
        if c == 0:
            u.likes = 0
        else:
            u.likes = likes/c

    @staticmethod
    def json_to_user(json):
        #print json
        u = user.User(json['uid'], json['first_name'], json['last_name'])
        u.sex = json.get('sex')
        u.set_age(VkApi.parse_birth_date(json.get('bdate')))

        u.university = json.get('university_name')

        if json.get('deactivated'):
            u.is_user_active = False
            return u
        counters = json.get('counters')
        try:
            g = counters.get('groups')
            if g is not None:
                u.groups = g
        except AttributeError:
            print "No attr GROUPS"
            time.sleep(2)

        try:
            ph = counters.get('user_photos')
            if ph is not None:
                u.user_photos = ph
        except AttributeError:
            print "No attr U_PHOTOS"
            time.sleep(2)

        try:
            vd = counters.get('user_videos')
            if vd is not None:
                u.user_videos = vd
        except AttributeError:
            print "No attr U_VIDEOS"
            time.sleep(2)

        rel = json.get('relation')
        if rel is not None:
            u.relation = rel
        u.status_len = len(json.get('status'))
        u.videos = counters.get('videos')
        u.notes = counters.get('notes')
        u.subscriptions = counters.get('subscriptions')
        u.mutual_friends = counters.get('mutual_friends')
        u.audios = counters.get('audios')
        u.photos = counters.get('photos')
        u.followers = counters.get('followers')
        u.albums = counters.get('albums')
        u.friends = counters.get('friends')
        u.pages = counters.get('pages')
        grad = json.get('graduation')
        if grad is not None:
            u.graduation = grad
        return u

    @staticmethod
    def parse_birth_date(birth_date_str):
        if birth_date_str:
            parts = birth_date_str.split('.')
            if len(parts) == 3:
                return datetime.date(int(parts[2]), int(parts[1]), int(parts[0]))


def main():
#    oauth.vk.com/authorize?client_id=4230129&scope=friends,photos&redirect_uri=https://oauth.vk.com/blank.html&display=\
#        page&v=5.16&response_type=token
    token = "628c9acb112579d0320e2dcdc145b7dfedb92b61b482592ff63a03fcb0066c7b6b91740825f1f88ad6bd7"
    api = VkApi(token)
    uids = api.get_friend_ids("88594620")

    f = open('monty_python.csv', 'w')
    cfw = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)

    for u in api.get_all_users(uids):
        print u.to_tsv()
        if u.is_user_active:
            cfw.writerow(u.to_file())

    f.close()


if __name__ == "__main__":
    main()