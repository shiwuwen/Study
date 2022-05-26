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


// 2、显示联系人
void showPerson(AddressBooks * abs){
    if(abs->m_Size == 0){
        cout << "通讯录为空" << endl;
    }
    else{
        for(int i = 0; i < abs->m_Size; i++){
            cout << "姓名： " << abs->personArray[i].m_Name << "\t";
            cout << "性别： " << abs->personArray[i].m_Sex << "\t";
            cout << "年龄： " << abs->personArray[i].m_Age << "\t";
            cout << "电话： " << abs->personArray[i].m_Phone << "\t";
            cout << "住址： " << abs->personArray[i].m_Addr << endl;;
        }
    }
}

// 判断联系人是否存在
int isExist(AddressBooks * abs, string name){
    for(int i = 0; i < abs->m_Size; i++){
        if(abs->personArray[i].m_Name == name){
            return i;
        }
    }

    return -1;
}


// 3、删除联系人
void deletePerson(AddressBooks * abs, string name){
    int ret = isExist(abs, name);

    if(ret == -1){
        cout << "查无此人" << endl;
    }
    else{
        abs->personArray[ret].m_Name = abs->personArray[abs->m_Size-1].m_Name;
        abs->personArray[ret].m_Sex = abs->personArray[abs->m_Size-1].m_Sex;
        abs->personArray[ret].m_Age = abs->personArray[abs->m_Size-1].m_Age;
        abs->personArray[ret].m_Phone = abs->personArray[abs->m_Size-1].m_Phone;
        abs->personArray[ret].m_Addr = abs->personArray[abs->m_Size-1].m_Addr;

        abs->m_Size--;

        cout << "删除成功" << endl;
    }
}


// 4、查找联系人
void findPerson(AddressBooks * abs, string name){
    int ret = isExist(abs, name);

    if(ret == -1){
        cout << "查无此人" << endl;
    }
    else{
        cout << "姓名： " << abs->personArray[ret].m_Name << "\t";
        cout << "性别： " << abs->personArray[ret].m_Sex << "\t";
        cout << "年龄： " << abs->personArray[ret].m_Age << "\t";
        cout << "电话： " << abs->personArray[ret].m_Phone << "\t";
        cout << "住址： " << abs->personArray[ret].m_Addr << endl;;
    }
}


// 5、修改联系人
void modifyPerson(AddressBooks * abs, string name){

}


// 6、清空联系人
void emptyPerson(AddressBooks * abs){
    abs->m_Size = 0;
    cout << "清空成功" << endl;
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
                showPerson(&abs);
                break;
            case 3: // 3、删除联系人
            {
                cout << "请输入待删除联系人姓名" << endl;
                string name;
                cin >> name;
                deletePerson(&abs, name);
            }
                break;
            case 4: // 4、查找联系人
            {
                cout << "请输入待查找联系人姓名" << endl;
                string name;
                cin >> name;
                findPerson(&abs, name);
            }
                break;
            case 5: // 5、修改联系人
                break;
            case 6: // 6、清空联系人
                emptyPerson(&abs);
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