def main() raises:
    log_events("User logged in", "File uploaded", "Error occurred", "User logged out")

def log_events(*events : String) raises:
    var Count : Int = 1
    for event in events:
        print("[Event {}] : {}".format(Count, event))
        Count += 1
    print("Total events logged:", Count - 1)