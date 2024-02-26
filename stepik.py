# s = '1:men 2:kind 90:number 0:sun 34:book 56:mountain 87:wood 54:car 3:island 88:power 7:box 17:star 101:ice'
# s = [_.split(':') for _ in s.split()]
# result = dict(s)

# # for i in range(len(s)):
# #     n = 
# result = {int(key): value for i in s.split() for key, value in [i.split(":")]}


# # result = {key[i] for key in range(len(s))}
# print(result)



tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15), (16, 17, 18), (19, 20, 21), (22, 23, 24), (25, 26, 27), (28, 29, 30), (31, 32, 33), (34, 35, 36)]

result = {key[0]: (key[1], key[2]) for key in tuples}
print(result)