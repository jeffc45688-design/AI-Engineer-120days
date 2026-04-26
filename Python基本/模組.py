#載入內建的 sys 模組並取得資訊
import sys as s
print(s.platform)
print(s.maxsize)
# #建立 geometry 模組,載入並使用
import geometry
result=geometry.slope(1,2,5,6)
print(result)
result=geometry.distance(1,1,5,5)
print(result)
#調整搜尋模組的路徑
import sys
sys.path.append("modules")#加入模組路徑
# print(sys.path)#顯示目前模組搜尋路徑
import geometry
print(geometry.distance(2,2,6,6))