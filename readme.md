# Генерация контента

Сервис генерирует статью на основе предоставленной темы и ссылок на актуальные веб-ресурсы.

## Начало работы

- Создайте и активируйте виртуальное окружение:  
`python3 -m venv venv`  
`source venv/bin/activate`
- Установите зависимости:  
`pip install -r requirements.txt`
- Создайте файл `.env`, в котором укажите:  
`OPENAI_API_KEY=<ваш ключ к api chat gpt>`
- Запустите проект, указав в аргументе `-u` ссылки на оригинальные статьи, а в аргументе `-q` название темы.  
Каждая ссылка должна быть заключена в кавычки и разделена пробелом.  
Тема также должна быть заключена в кавычки.  
`python main.py -u "<ссылка>" "<ссылка>" ... "<ссылка>" -q "<название темы>"`

## Примеры

- **Тема**: "ТОП 5 трав для выращивания на подоконнике. Как стать садоводом в тренде дома?"  
**Запрос**:
```
python main.py -u "https://www.7ya.ru/article/Chto-mozhno-vyrastit-v-kachestve-mikrozeleni/?utm_source=yxnews&utm_medium=desktop&utm_referrer=https%3A%2F%2Fdzen.ru%2Fnews%2Fsearch%3Ftext%3D" "https://www.agroxxi.ru/zhurnal-agromir-xxi/stati-rastenievodstvo/tradeskancija-lechebnye-svoistva-i-vyraschivanie-doma.html?utm_source=yxnews&utm_medium=desktop&utm_referrer=https%3A%2F%2Fdzen.ru%2Fnews%2Fsearch%3Ftext%3D" "https://www.advis.ru/php/view_news.php?id=9E737153-8ADC-0B4B-96F6-83CBCB9415FC&utm_source=yxnews&utm_medium=desktop&utm_referrer=https%3A%2F%2Fdzen.ru%2Fnews%2Fsearch%3Ftext%3D" "https://www.kirov.kp.ru/daily/27580.5/4849898/?utm_source=yxnews&utm_medium=desktop&utm_referrer=https%3A%2F%2Fdzen.ru%2Fnews%2Fsearch%3Ftext%3D" "https://riamo.ru/article/662765/kazhdyj-vtoroj-zhitel-goroda-v-rf-vyraschivaet-na-podokonnike-ovoschi-i-zelen-opros?utm_source=yxnews&utm_medium=desktop&utm_referrer=https%3A%2F%2Fdzen.ru%2Fnews%2Fsearch%3Ftext%3D" "https://aif.ru/dacha/ogorod/ogorod_zimoy_v_rossii_nashli_sposob_sekonomit_na_zeleni_i_ovoshchah?utm_source=yxnews&utm_medium=desktop&utm_referrer=https%3A%2F%2Fdzen.ru%2Fnews%2Fsearch%3Ftext%3D" "https://inmozhaisk.ru/news/ekologiya/vitaminy-na-okne-sadovod-iz-mozhajska-rasskazala-kak-vyrastit-zelen-v-kvartire?utm_source=yxnews&utm_medium=desktop&utm_referrer=https%3A%2F%2Fdzen.ru%2Fnews%2Fsearch%3Ftext%3D" "https://zelen-na-podokonnike.ru/" -q "ТОП 5 трав для выращивания на подоконнике. Как стать садоводом в тренде дома?"
```

**Ответ**:  
"""  
ТОП 5 трав для выращивания на подоконнике. Как стать садоводом в тренде дома?

Выращивание своих собственных трав на подоконнике становится все более популярным среди людей, которые хотят иметь доступ к свежим и ароматным травам прямо у себя дома. Это не только позволяет сэкономить деньги на покупке трав в магазине, но и дает возможность контролировать качество и свежесть продукта. В этой статье мы рассмотрим ТОП 5 трав, которые легко вырастить на подоконнике, и поделимся советами о том, как стать садоводом в тренде дома.

1. Базилик
Базилик - это одна из самых популярных трав, которую можно выращивать на подоконнике. Он обладает ярким ароматом и вкусом, и идеально подходит для приготовления итальянских блюд, салатов и соусов. Базилик требует яркого освещения и умеренного полива. Вы можете начать с покупки готовой рассады или посадить семена в горшок с плодородной почвой. Через несколько недель вы сможете наслаждаться свежим базиликом прямо с подоконника.

2. Руккола
Руккола - это зелень с нежными листьями и острым вкусом. Она идеально подходит для салатов, сэндвичей и пасты. Руккола легко выращивается на подоконнике и требует яркого освещения. Вы можете посадить семена в горшок с плодородной почвой и через несколько недель наслаждаться свежей рукколой.

3. Петрушка
Петрушка - это одна из самых популярных трав, которую можно выращивать на подоконнике. Она обладает свежим ароматом и вкусом, и идеально подходит для приготовления супов, соусов и салатов. Петрушка требует яркого освещения и умеренного полива. Вы можете посадить семена в горшок с плодородной почвой и через несколько недель наслаждаться свежей петрушкой.

4. Укроп
Укроп - это трава с ярким ароматом и вкусом, которая идеально подходит для приготовления салатов, соусов и маринадов. Укроп требует яркого освещения и регулярного полива. Вы можете посадить семена в горшок с плодородной почвой и через несколько недель наслаждаться свежим укропом.

5. Мята
Мята - это трава с освежающим ароматом и вкусом, которая идеально подходит для приготовления чая, коктейлей и десертов. Мята требует яркого освещения и умеренного полива. Вы можете посадить семена в горшок с плодородной почвой и через несколько недель наслаждаться свежей мятой.

Теперь, когда вы знаете ТОП 5 трав для выращивания на подоконнике, давайте рассмотрим несколько советов о том, как стать садоводом в тренде дома.

1. Выберите подходящие горшки и почву. Горшки должны иметь отверстия для дренажа, чтобы избежать застоя воды. Почва должна быть плодородной и хорошо дренированной.

2. Обеспечьте достаточное освещение. Травы требуют яркого освещения, поэтому разместите горшки на самом светлом месте на подоконнике или используйте специальные светодиодные лампы для растений.

3. Регулярно поливайте травы. Умеренный полив поможет поддерживать оптимальный уровень влажности почвы.

4. Удобряйте травы. Регулярное удобрение поможет поддерживать здоровый рост и развитие трав.

5. Подрезайте травы. Регулярная обрезка поможет поддерживать компактную форму и стимулировать рост новых побегов.

Следуя этим советам, вы сможете стать садоводом в тренде дома и наслаждаться свежими и ароматными травами прямо у себя на подоконнике.  
"""

</br>

- **Тема**: "Сигнализация для дачи: как выбрать и установить? Как узнать, что в доме что-то произошло?"  
**Запрос**:  
```
python main.py -u "https://www.kp.ru/expert/dom/luchshie-okhrannye-sistemy-dlya-doma/" "https://ps-link.ru/catalog/signalizatsii/okhrannye_signalizatsii/gsm_cignalizatsii_strazh_video/578/" "https://vashumnyidom.ru/oxrana/ustanovka-oxrannoi-signalizacii.html#montazh-signalizatsii" -q "Сигнализация для дачи: как выбрать и установить? Как узнать, что в доме что-то произошло?"
```
**Ответ**:  
"""  
Сигнализация для дачи: как выбрать и установить? Как узнать, что в доме что-то произошло?

Сигнализация для дачи является важным элементом безопасности и защиты вашего имущества. Ведь дача, как правило, находится в удаленном месте, где отсутствует постоянное присутствие людей. Поэтому необходимо обеспечить надежную защиту от воровства и возможных ЧП.

Выбор и установка сигнализации для дачи требует определенных знаний и внимания к деталям. Важно учесть особенности местности, размеры дачного участка, а также ваши потребности и бюджет.

Перед выбором сигнализации для дачи, рекомендуется определиться с несколькими ключевыми вопросами:

1. Тип сигнализации: существуют различные типы сигнализации, такие как проводная, беспроводная, GSM-сигнализация и системы "умного дома". Каждый тип имеет свои преимущества и недостатки, поэтому важно выбрать тот, который наилучшим образом соответствует вашим потребностям.

2. Комплектация: определите, какие датчики и устройства вам необходимы. Например, датчики движения, датчики открытия дверей и окон, датчики протечки воды и дыма. Выберите комплектацию, которая наиболее эффективно защитит вашу дачу от различных угроз.

3. Управление и оповещение: удобство управления и оповещения являются важными аспектами сигнализации для дачи. Рассмотрите возможность управления сигнализацией через мобильное приложение, SMS-сообщения или дистанционное управление. Также убедитесь, что система оповещения надежна и эффективна.

4. Установка: для установки сигнализации на даче можно обратиться к специалистам или сделать это самостоятельно. Если вы решите установить сигнализацию самостоятельно, важно следовать инструкциям производителя и учесть особенности местности и конструкции дома.

Как узнать, что в доме что-то произошло? Современные сигнализации для дачи обеспечивают различные способы оповещения о происходящих событиях. Это может быть SMS-сообщение, звонок на мобильный телефон, уведомление через мобильное приложение или даже голосовое сообщение. Выберите тот способ оповещения, который наиболее удобен для вас и обеспечивает быструю реакцию на происходящее.

В заключение, выбор и установка сигнализации для дачи являются важными шагами для обеспечения безопасности вашего имущества. При выборе сигнализации учитывайте особенности местности, потребности и бюджет. Установка сигнализации может быть выполнена самостоятельно или с помощью специалистов. Важно следовать инструкциям производителя и обеспечить надежное оповещение о происходящих событиях.  
"""

</br>

- **Тема**: "Осенняя кладезь витаминов: ценность клюквы, облепихи и необычные рецепты"
**Запрос**:
```
python main.py -u "https://www.edimdoma.ru/jivem_doma/posts/22376-vse-na-borbu-s-avitaminozom-vkusnye-i-poleznye-yagody-v-osennem-menyu" "https://www.sadurala.com/blog/lovkost-ruk-i-nikakih-zabot-kak-sobirat-oblepihu" "https://lefood.menu/recipes/bedryshki-indeyki-zapechennye-s-chesnokom-apelsinami-i-klyukvoy/" "https://elementaree.ru/blog/science/klyukva-polza-i-vred/" "https://www.iamcook.ru/showrecipe/13639" "https://www.russianfood.com/recipes/recipe.php?rid=153501" "https://www.rigla.ru/about/news/2022/oblepiha" -q "Осенняя кладезь витаминов: ценность клюквы, облепихи и необычные рецепты"
```
**Ответ**:  
"""  
Осенняя кладезь витаминов: ценность клюквы, облепихи и необычные рецепты

Осень - это время, когда природа щедро дарит нам разнообразные ягоды, которые являются настоящей кладезью витаминов и минералов. Среди этих ягод особое место занимают клюква и облепиха. Они не только богаты полезными веществами, но и имеют множество необычных рецептов приготовления.

Клюква - это ягода, которая обладает ярким кисло-сладким вкусом и ароматом. Она богата витаминами С, А, Е, а также микроэлементами, такими как калий, кальций, магний и железо. Клюква имеет мощное антиоксидантное действие, способствует укреплению иммунитета и защите организма от вредных воздействий окружающей среды. Кроме того, клюква полезна для пищеварения, она способствует улучшению работы желудка и кишечника.

Облепиха - это ягода, которая известна своим уникальным составом. Она богата витаминами А, С, Е, а также микроэлементами, такими как калий, кальций, магний и железо. Облепиха имеет мощное противовоспалительное и ранозаживляющее действие, способствует укреплению иммунитета и защите организма от вредных воздействий окружающей среды. Кроме того, облепиха полезна для зрения, улучшает состояние кожи, волос и ногтей.

Теперь давайте рассмотрим несколько необычных рецептов приготовления блюд с использованием клюквы и облепихи.

1. Клюквенный соус для мяса. Для приготовления этого соуса вам понадобятся свежие ягоды клюквы, сахар, вода и крахмал. Сначала вымойте ягоды клюквы и протрите их через сито, чтобы получить пюре. Затем смешайте пюре с сахаром и водой, и варите на медленном огне до загустения. Готовый соус можно подавать к жареному или запеченному мясу.

2. Облепиховый пирог. Для приготовления этого пирога вам понадобятся ягоды облепихи, мука, сахар, яйца, сливочное масло и шоколад. Сначала приготовьте основу для пирога, смешав муку, сахар и сливочное масло. Затем выложите основу в форму и выпекайте в духовке до готовности. Пока основа остывает, приготовьте начинку из ягод облепихи, сахара и шоколада. Вылейте начинку на основу и поставьте в холодильник до застывания. Пирог готов к подаче.

3. Клюквенный мусс. Для приготовления этого мусса вам понадобятся свежие ягоды клюквы, сахар, вода, манка, мед и сливочное масло. Сначала приготовьте ягодное пюре, протерев ягоды через сито. Затем смешайте пюре с сахаром, водой и манкой, и варите на медленном огне до загустения. После этого добавьте мед и сливочное масло, и остудите мусс в холодильнике. Готовый мусс можно подавать к чаю или использовать в качестве начинки для пирогов.

Таким образом, клюква и облепиха - это настоящая осенняя кладезь витаминов. Они не только богаты полезными веществами, но и имеют множество необычных рецептов приготовления. Попробуйте приготовить блюда с использованием этих ягод и насладитесь их вкусом и пользой для здоровья.  
"""