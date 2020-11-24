import os

leaf_origin = [
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,170,174,175,176,177,178,
    179,180,181,182,183,184,185,186,187,
    188,189,190,191,192,193,194,195,196,
    197,198,199,200,201,202,203,204,205,
    206,207,208,209,210,211,231,446,565,
    568,571,573,578,581,582,583,587,600,
    601,602,603,612,613,614,616,618,624,
    625,626,627,629,630,631,633,635,642,
    643,649,656,667,674,684,685,687,693,
    694,696,701,703,707,708,727,734,736,
    740,741,745,747,750,752,758,759,763,
    766,773,775,776,778,780,781,789,792,
    793,803,810,815,816,824,825,826,827,
    830,831,833,837,841,842,845,848,852,
    853,854,866,877,879,884,886,891,904,
    909,911,913,915,919,921,924,933,934,
    935,936,937,938,939,940,941,942,943,
    944,945,946,947,948,949,950,951,952,
    953,954,955,956,957,958,959,960,961,
    962,963,964,965,966,967,968,969,970,
    971,972,973,974,975,976,977,978,979,
    980,981,982,983,984,985,986,987,988,
    989,990,991,992,993,994,995,998,999,
    1001,1006,1007,1037,1047,1048,1049,
    1054,1066,1069,1071,1076,1081,1084,
    1089,1093,1105,1110,1115,1116,1118,
    1124,1127,1134,1151,1152,1156,1164,
    1169,1180,1181,1183,1184,1187,1193,
    1194,1196,1198,1201,1218,1226,1228,
    1229,1230,1370,1624,1683,1685,1742
]

dir_origin = './leaf'
dir_target = './leaf_0'

for sample in leaf_origin:
    os.replace(os.path.join(dir_origin, str(sample)+'.jpg'), os.path.join(dir_target, str(sample)+'.jpg'))
    print (os.path.join(dir_origin, str(sample)+'.jpg'))


print (len(leaf_origin))

