#include <iostream>
#include <string>
#define MAX 1000
using namespace std;

// 显示菜单
void showMenu(){

    cout << "**************************" << endl;
    cout << "*****  1、添加联系人  *****" << endl;
    cout << "*****  2、显示联系人  *****" << endl;
    cout << "*****  3、删除联系人  *****" << endl;
    cout << "*****  4、查找联系人  *****" << endl;
    cout << "*****  5、修改联系人  *****" << endl;
    cout << "*****  6、清空联系人  *****" << endl;
    cout << "*****  0、退出系统    *****" << endl;
    cout << "**************************" << endl;
}

// 设计结构体
struct Person {

    string m_Name;

    int m_Sex;

    int m_Age;

    string m_Phone;

    string m_Addr;
};

struct AddressBooks{

    Person personArray[MAX];

    int m_Size;
};

// 1、添加联系人
void addPerson(AddressBooks * abs){

    if(abs->m_Size == MAX){
        cout << "通讯录已满" << endl;
        return;
    }
    else{

        // 添加姓名
        string name;
        cout << "请输入姓名" << endl;
        cin >> name;
        abs->personArray[abs->m_Size].m_Name = name;

        // 性别
        int sex;
        cout << "请输入性别" << endl;
        cout << "1 --- 男" << endl;
        cout << "2 --- 女" << endl;

        while(true){
            cin >> sex;

            if(sex == 1 || sex == 2){
                abs->personArray[abs->m_Size].m_Sex = sex;
                break;
            }
            cout << "输入有误" << endl;
        }

        // 年龄
        int age;
        cout << "请输入年龄" << endl;
        cin >> age;
        abs->personArray[abs->m_Size].m_Age = age;

        //电话
        string phoneNum;
        cout << "请输入电话" << endl;
        cin >> phoneNum;
        abs->personArray[abs->m_Size].m_Phone = phoneNum;

        //地址
        string address;
        cout << "请输入地址" << endl;
        cin >> address;
        abs->personArray[abs->m_Size].m_Addr = address;

        abs->m_Size++;
        cout << "添加成功" << endl;
        
    }
}

int main(){
    int select = 0;

    AddressBooks abs;
    abs.m_Size = 0;

    while(true){

        showMenu();
        cin >> select;

        switch(select){
            case 1: // 1、添加联系人
                addPerson(&abs);
                break;
            case 2: // 2、显示联系人
                break;
            case 3: // 3、删除联系人
                break;
            case 4: // 4、查找联系人
                break;
            case 5: // 5、修改联系人
                break;
            case 6: // 6、清空联系人
                break;
            case 0: // 0、退出系统
                cout << "欢迎下次使用！" << endl;
                return 0;
                break;
            default:
                cout << "输入的内容无效" << endl;
                break;
        }
    }

    return 0;
}