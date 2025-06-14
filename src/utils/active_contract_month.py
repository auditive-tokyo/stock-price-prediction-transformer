import datetime

def determine_active_contract_month(expiration_dates_list, current_system_date):
    """
    Determines the active futures contract month based on a list of expiration dates.

    Args:
        expiration_dates_list (list): A list of expiration date strings,
                                      expected in "YYYYMMDD" format and sorted.
        current_system_date (datetime.date): The current date to compare against.

    Returns:
        str: The selected contract month in "YYYYMM" format, or None if no suitable
             contract is found.
    """
    selected_yyyymm = None
    print(f"\nDetermining active contract based on current date: {current_system_date}")
    print(f"Available expiration dates (YYYYMMDD): {expiration_dates_list}")

    for date_str in expiration_dates_list:
        if len(date_str) == 8 and date_str.isdigit():
            try:
                exp_year = int(date_str[:4])
                exp_month = int(date_str[4:6])
                exp_day = int(date_str[6:8])
                expiration_date_obj = datetime.date(exp_year, exp_month, exp_day)

                # We need the contract that is still active or the next one.
                # If the last trading day (expiration_date_obj) is after the current_system_date,
                # it's a candidate. The first such candidate in a sorted list is the one.
                if expiration_date_obj > current_system_date:
                    selected_yyyymm = f"{exp_year:04d}{exp_month:02d}"
                    print(f"Selected: {selected_yyyymm} (from expiration date {date_str}) as it's after {current_system_date}")
                    break  # Found the first suitable contract
            except ValueError:
                print(f"Warning: Could not parse date string '{date_str}' as YYYYMMDD.")
        else:
            print(f"Warning: Unexpected expiration date format '{date_str}'. Expected YYYYMMDD.")

    if not selected_yyyymm:
        print("No suitable active or upcoming futures contract month found.")
    
    return selected_yyyymm