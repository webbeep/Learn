"""Synthetic 'tiny world' corpus.

We generate simple stories on purpose: the patterns are learnable by a ~200k
parameter model in a few minutes on a laptop CPU, and they are easy to *read*
in the visualizations (names get copied, colors stick to objects, days follow
days, questions get answered).
"""
import random
import re

NAMES = ["mia", "tom", "lily", "sam", "ben", "zoe", "max", "ava", "leo", "ruby", "finn", "nora"]
ANIMALS = ["cat", "dog", "fox", "bird", "frog", "bear", "owl", "fish", "duck", "rabbit"]
OBJECTS = ["ball", "kite", "hat", "book", "cup", "box", "drum", "shoe", "boat", "key"]
COLORS = ["red", "blue", "green", "yellow", "pink", "black", "white", "orange", "purple"]
PLACES = ["park", "forest", "river", "school", "garden", "beach", "house", "hill"]
FOODS = ["apple", "cake", "bread", "soup", "cookie", "pear", "honey", "cheese"]
VERBS = [("jump", "jumped", "jumping"), ("play", "played", "playing"), ("walk", "walked", "walking"),
         ("look", "looked", "looking"), ("climb", "climbed", "climbing"), ("laugh", "laughed", "laughing"),
         ("dance", "danced", "dancing"), ("sing", "sang", "singing"), ("run", "ran", "running")]
FEEL = ["happy", "sad", "tired", "hungry", "excited", "sleepy", "brave"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NUMS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
FACTS = [("sky", "blue"), ("grass", "green"), ("snow", "white"), ("sun", "hot"), ("ice", "cold"),
         ("night", "dark"), ("sugar", "sweet"), ("lemon", "sour"), ("sea", "big"), ("ant", "small")]
SOUNDS = {"cat": "meow", "dog": "woof", "duck": "quack", "owl": "hoot", "frog": "ribbit", "bear": "grr",
          "bird": "tweet", "fox": "yip", "fish": "blub", "rabbit": "squeak"}


def pronoun(name):
    return "she" if name in ("mia", "lily", "zoe", "ava", "ruby", "nora") else "he"


def trade(r):
    a, b = r.sample(NAMES, 2)
    o1, o2 = r.sample(OBJECTS, 2)
    c1, c2 = r.sample(COLORS, 2)
    return (f"{a} has a {c1} {o1}. {b} has a {c2} {o2}. {a} gives the {o1} to {b}. "
            f"now {b} has the {c1} {o1}.")


def pet(r):
    n = r.choice(NAMES); an = r.choice(ANIMALS); c = r.choice(COLORS); p = pronoun(n)
    return (f"{n} has a {c} {an}. the {an} says {SOUNDS[an]}. {p} loves the {an}. "
            f"the {an} is {c}.")


def trip(r):
    n = r.choice(NAMES); pl = r.choice(PLACES); v = r.choice(VERBS); f = r.choice(FEEL); p = pronoun(n)
    food = r.choice(FOODS)
    return (f"{n} went to the {pl}. at the {pl}, {p} {v[1]} all day. "
            f"{p} was {v[2]} and {f}. then {n} ate a {food}.")


def story(r):
    an = r.choice(ANIMALS); n = r.choice(NAMES); pl = r.choice(PLACES); c = r.choice(COLORS)
    o = r.choice(OBJECTS); f = r.choice(FEEL)
    return (f"once upon a time, there was a little {an} named {n}. {n} lived near the {pl}. "
            f"one day, {n} found a {c} {o}. {n} was so {f}! the end.")


def days(r):
    i = r.randrange(7); k = r.randint(2, 5)
    seq = " ".join(DAYS[(i + j) % 7] for j in range(k))
    return f"{seq}. after {DAYS[(i + k - 1) % 7]} comes {DAYS[(i + k) % 7]}."


def count(r):
    i = r.randrange(5); k = r.randint(3, 5)
    return " ".join(NUMS[i:i + k]) + "."


def fact(r):
    t, a = r.choice(FACTS)
    return r.choice([f"the {t} is {a}.", f"q: what is the {t} like? a: the {t} is {a}."])


def qa(r):
    n = r.choice(NAMES); o = r.choice(OBJECTS); c = r.choice(COLORS)
    return f"{n} has a {c} {o}. q: what color is the {o}? a: the {o} is {c}."


def sound_q(r):
    an = r.choice(ANIMALS)
    return f"q: what does the {an} say? a: the {an} says {SOUNDS[an]}."


GENS = [(trade, 3), (pet, 2), (trip, 2), (story, 3), (days, 1), (count, 1), (fact, 1), (qa, 3), (sound_q, 1)]


def make_corpus(n_docs=60000, seed=0):
    r = random.Random(seed)
    fns = [f for f, w in GENS for _ in range(w)]
    text = "\n".join(r.choice(fns)(r) for _ in range(n_docs)) + "\n"
    return re.sub(r"\b(a) ([aeiou])", r"an \2", text)


if __name__ == "__main__":
    print(make_corpus(8, seed=1))
