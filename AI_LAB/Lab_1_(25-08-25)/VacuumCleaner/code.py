def vacuum_cleaner_agent():

    state = {
        'A': input("Enter status of location A (0=Clean, 1=Dirty): ").strip(),
        'B': input("Enter status of location B (0=Clean, 1=Dirty): ").strip(),
        'vacuum_pos': input("Enter vacuum start location (A or B): ").strip().upper()
    }

    if state['vacuum_pos'] not in ('A', 'B'):
        print("Invalid vacuum start location. Must be 'A' or 'B'.")
        return
    if state['A'] not in ('0', '1') or state['B'] not in ('0', '1'):
        print("Invalid status for locations. Must be '0' or '1'.")
        return

    cost = 0

    def is_dirty(location):
        return state[location] == '1'

    def clean(location):
        nonlocal cost
        if is_dirty(location):
            print(f"Cleaning location {location}.")
            state[location] = '0'
            cost += 1 
        else:
            print(f"Location {location} already clean.")

    def move_to(location):
        nonlocal cost
        if state['vacuum_pos'] != location:
            print(f"Moving vacuum from {state['vacuum_pos']} to {location}.")
            state['vacuum_pos'] = location
            cost += 1  
        else:
            print(f"Vacuum already at location {location}.")

    print("\nStarting Vacuum Cleaner Agent\n")
    print(f"Initial State: A={state['A']}, B={state['B']}, Vacuum at {state['vacuum_pos']}\n")

    while state['A'] == '1' or state['B'] == '1':
        current_loc = state['vacuum_pos']
        clean(current_loc)

        other_loc = 'B' if current_loc == 'A' else 'A'
        if is_dirty(other_loc):
            move_to(other_loc)
        else:
            if state['A'] == '0' and state['B'] == '0':
                print("Both locations clean. Task completed.")
                break
            else:
                break

    print("\nFinal State:")
    print(f"Location A: {'Clean' if state['A']=='0' else 'Dirty'}")
    print(f"Location B: {'Clean' if state['B']=='0' else 'Dirty'}")
    print(f"Vacuum Location: {state['vacuum_pos']}")
    print(f"Total cost (actions taken): {cost}")

if __name__ == "__main__":
    vacuum_cleaner_agent()

