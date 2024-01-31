s = '1:men 2:kind 90:number 0:sun 34:book 56:mountain 87:wood 54:car 3:island 88:power 7:box 17:star 101:ice'
s = [_.split(':') for _ in s.split()]
result = dict(s)

# for i in range(len(s)):
#     n = 
result = {int(key): value for i in s.split() for key, value in [i.split(":")]}


# result = {key[i] for key in range(len(s))}
print(result)