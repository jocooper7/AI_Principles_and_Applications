# ============================================================
# Function: process_employee_data(emp_id, hours_worked, pay_rate, dept)
# Purpose: Calculate an employee's total pay, overtime, bonus,
#          taxes, and net pay, then log and return a payment
#          record for the given employee.
# ============================================================
def process_employee_data(emp_id, hours_worked, pay_rate, dept):
    total_pay = hours_worked * pay_rate
    
    employee_name = str(emp_id) + " Employee"
    
    print("Starting calculation for " + employee_name)
    
    if hours_worked > 40:
        overtime_hours = hours_worked - 40
        overtime_pay = overtime_hours * pay_rate * 1.5
    else:
        overtime_pay = 0
    
    TotalPay = total_pay + overtime_pay
    
    log_entry = employee_name + str(TotalPay) + " calculated"
    
    if dept == "Sales":
        bonus = TotalPay * 0.1
    elif dept == "IT":
        bonus = TotalPay * 0.05
    else:
        bonus = 0
    
    final_salary = TotalPay + bonus
    
    if dept == "Sales":
        dept_index = 0
    elif dept == "IT":
        dept_index = 1
    else:
        dept_index = 2
    
    dept_codes = ["S", "I", "M"]
    dept_code = dept_codes[dept_index]
    
    if final_salary > 0:
        payment_status = "Due"
    else:
        payment_status = "Not Due"
    
    tax_rate = 0.25
    taxes = final_salary * tax_rate
    
    net_pay = final_salary - taxes
    
    with open("log.txt", "w") as file:
        file.write(log_entry)
    
    result = {
        "id": emp_id,
        "pay": final_salary,
        "net_pay": net_pay,
        "dept": dept_code,
        "status": payment_status
    }
    
    return result

print(process_employee_data(1001, 40, 15.0, "Sales"))