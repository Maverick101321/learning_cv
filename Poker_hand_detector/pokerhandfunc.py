def find_poker_hand(hand):
    ranks = []
    suits = []
    possible_ranks = []

    for card in hand:
        if len(card) == 2:
            rank = card[0]
            suit = card[1]
        else:
            rank = card[0:2]
            suit = card[2]
        #Defining the ranks value
        if rank == "A":
            rank = 14
        elif rank == "K":
            rank = 13
        elif rank == "Q":
            rank = 12
        elif rank == "J":
            rank = 11
        ranks.append(int(rank))
        suits.append(suit)
    #print(ranks)
    sorted_rank = sorted(ranks)
    #print(sorted_rank)

    #Royal Flush, Flush, Straight Flush
    #Checking if all of the cards are of the same suit (flush)
    if suits.count(suits[0]) == 5:
        if 14 in sorted_rank and 13 in sorted_rank and 12 in sorted_rank and 11 in sorted_rank \
            and 10 in sorted_rank:
            possible_ranks.append(10) #Royal Flush
        elif all(sorted_rank[i] == sorted_rank[i-1] + 1 for i in range(1, len(sorted_rank))):
            possible_ranks.append(9) #Straight Flush
        else:
            possible_ranks.append(6) #Flush

    #Straight
    #Checking if all the ranks are consecutive
    if all(sorted_rank[i] == sorted_rank[i-1] + 1 for i in range(1, len(sorted_rank))):
        possible_ranks.append(5)

    unique_hand_val = list(set(sorted_rank))
    
    # 3 3 3 3 5 = set = 3, 5 : Four of a Kind lenght = 2
    # 3 3 3 4 4 = set = 3, 4 : Full House lenght = 2

    #Four of a Kind and Full House
    if len(unique_hand_val) == 2:
        for val in unique_hand_val:
            if sorted_rank.count(val) == 4:
                possible_ranks.append(8)
            if sorted_rank.count(val) == 3:
                possible_ranks.append(7)

    
    # 5 5 5 8 9 = set = 5, 8, 9 : Three of a Kind lenght = 3
    # 5 5 8 8 9 = set = 5, 8, 9 : Two Pair lenght = 3

    #Three of a Kind and Two Pair
    if len(unique_hand_val) == 3:
        for val in unique_hand_val:
            if sorted_rank.count(val) == 3:
                possible_ranks.append(4)
            if sorted_rank.count(val) == 2:
                possible_ranks.append(3)

    #One Pair or Pair
    # 5 5 3 6 7 = set = 5, 3, 6, 7 : One Pair lenght = 4
    if len(unique_hand_val) == 4:
        possible_ranks.append(2)

    #For now returning all high cards if the hand is not a royal flush
    if not possible_ranks:
        possible_ranks.append(1)


    poker_hand_ranks = {10: "Royal Flush", 9: "Straight Flush", 8: "Four of a Kind", 7: "Full House", 
                        6: "Flush", 5: "Straight", 4: "Three of a Kind", 3: "Two Pair", 2: "One Pair", 
                        1: "High Card"}
    
    #print(hand)
    output = poker_hand_ranks[max(possible_ranks)]
    print(hand, output)
    return output

if __name__ == "__main__":
    find_poker_hand(["AH", "KH", "QH", "JH", "10H"]) #Royal Flush
    find_poker_hand(["QC", "JC", "10C", "9C", "8C"]) #Straight Flush
    find_poker_hand(["5C", "5S", "5H", "5D", "QH"]) #Four of a Kind
    find_poker_hand(["2H", "2D", "2S", "10H", "10C"]) #Full House
    find_poker_hand(["2D", "KD", "7D", "6D", "5D"]) #Flush
    find_poker_hand(["JC", "10H", "9C", "8C", "7D"]) #Straight
    find_poker_hand(["10H", "10C", "10D", "2D", "5S"]) #Three of a Kind
    find_poker_hand(["KD", "KH", "5C", "5S", "6D"]) #Two Pair
    find_poker_hand(["2D", "2S", "9C", "KD", "10C"]) #Pair
    find_poker_hand(["KD", "5H", "2D", "10C", "JH"]) #High Card