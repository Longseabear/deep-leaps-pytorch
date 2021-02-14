import collections
try:
    from collections.abc import MutableSet
except ModuleNotFoundError:
    from collections import MutableSet

class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

    def update(self, data):
        for i in data:
            self.add(i)
if __name__ == '__main__':
    s = OrderedSet([1,2,3,4,5])
    t = OrderedSet([1,2,3,4,5])
    print(s | t)
    print(s & t)
    print(s - t)
    print(s)
    for i in t:
        print(i)
    t.discard(3)
    print(t)
    t.add(3)
    print(t)
    t.add(3)
    print(t)
    t.add(3)
    t.update([4,6,5,7,8])
    print(t)
    print(len(t))
