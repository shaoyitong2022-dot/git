def simulate_game():
    # This function simulates a simple game where a player rolls a die
    import random

    print("Welcome to the Dice Rolling Game!")
    input("Press Enter to roll the die...")

    # Roll the die
    die_roll = random.randint(1, 6)
    print(f"You rolled a {die_roll}!")

    # Determine the outcome based on the die roll
    if die_roll == 6:
        print("Congratulations! You win!")
    else:
        print("Sorry, you lose. Better luck next time!")