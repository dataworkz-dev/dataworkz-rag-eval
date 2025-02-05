from dataworkz.dtwz_ai import AIDtwz
from pprint import pprint


def main() -> None:
    dtwz_ai_client = AIDtwz()
    while True: 
        print("1. Get QNA system details")
        print("2. Get LLM Provider ID")
        print("3. Exit")
        choice = int(input("Enter your choice: "))
        if choice == 3:
            exit(0)
        elif choice == 1:
            print("QNA system details:\n")
            pprint(dtwz_ai_client.get_qna_systems())
        elif choice == 2:
            qna_system_id = input("Enter QNA system ID: ")
            dtwz_ai_client.set_system_id(qna_system_id)
            print(f"LLM Provider IDs for {qna_system_id}:\n")
            pprint(dtwz_ai_client.get_llm_provider_details())
        else:
            print("Invalid choice.")
        print("\n")
    return


if __name__ == "__main__":
    main()