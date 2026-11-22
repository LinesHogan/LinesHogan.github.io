---
layout: post
title: "AI浪潮与逃难，还有实事求是"
date: 2026-11-22 13:00:00 +0800
description: "在算法、Infra 与业务之间跨界“逃难”，往往只是把外行的转换成本误当成了 AI 跨不过去的护城河。（支持 EN / 中文 一键切换，含 TL;DR）"
tags: ai-essays reflections career
categories: essays
related_posts: true
---

<div class="zine-dispatch" data-title-zh="AI浪潮与逃难，还有实事求是" data-title-en="The AI Wave, Seeking Refuge, and Seeking Truth from Facts">
  <a class="zine-pill-link" data-i18n-zh="同步发布于知乎 ↗" data-i18n-en="Cross-posted on Zhihu ↗" href="https://zhuanlan.zhihu.com/p/2107815235616883810" target="_blank" rel="noopener">同步发布于知乎 ↗</a>
  <div class="zine-lang-switch" role="group" aria-label="Language">
    <button type="button" class="zine-lang-btn" data-lang-target="en" aria-pressed="false">EN</button>
    <button type="button" class="zine-lang-btn is-active" data-lang-target="zh" lang="zh-CN" aria-pressed="true">中文</button>
  </div>
</div>

<div class="zine-lang-panel" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR：** 在算法、Infra 与业务之间跨界“逃难”，往往只是把外行的转换成本误当成了 AI 跨不过去的护城河。在资本主义激励机制与神经网络表达能力的双重驱动下，静态的技术壁垒大概率会被逐一填平；与其到处找避难所，不如靠过硬的产出、服务社区的基础建设，以及在普遍 overclaim 的行业里坚持“实事求是”，攒下低核验成本的信用，把自己绑上资源分配的战车。但更彻底的实事求是，是承认谁也算不准非平稳系统的终局——放下对几十年后安全身位的过度盘算，把眼下真正重要、做得够牛逼且对大家有用的事做好，大概就够了。

</div>

人们常常觉得呀，把还不熟悉的事物奉为圭臬、视作护城河，就能给自己找一个催眠的借口，说服自己这行还有上升空间，而且这空间是有门槛的，不至于轻易被摧毁。这种心理不止在技术行业里常见，放到国家与社会、学科与信仰上，也大抵如此。

但这种信念通常不太靠得住。这几年我见过无数原本做算法的人转去做 Infra，也见过大批做 Infra 的人回头去学算法。两边都误以为对方手里攥着某种硬门槛，而自己熟悉的一亩三分地正被 AI 蚕食，于是纷纷想靠跨界学习去“逃难”。至于算法和 Infra 两边都摸过的人呢？他们很清楚，无论是靠 Auto Research 调算法，还是做自动化的 Infra 优化，如今都没剩多少壁垒可言，于是又退而求其次，指望和现实世界沾边的复杂业务能充当护城河，帮自己躲过 AI 的追杀。每一拨人都能找出一套理由，论证自己新选的避难所和原先的领域是“本质不同的”。

这种靠宣布“本质不同”来给自己发免死金牌的做法，倒也算不上当代程序员的独创。战国末年秦军东出的时候，六国的士人大概也觉得齐鲁的礼乐传统与关中的耕战制度“本质不同”；古腾堡的铅字印刷机刚普及那会儿，中世纪的神学家们想来也坚信拉丁文的经院哲学自有其不可替代的门槛。人们总是很容易把“外行学起来费劲的转换成本”，当成“技术跨不过去的客观壁垒”。

---

说到底，“本质不同”本身就是个挺幽默的说法——谁又能说得清到底什么是本质呢？所谓本质，无非是由还原论的深度所决定的概念：你愿意往下还原几层，本质就可以停在哪一层。拿大模型来说，有人说本质是概率模型，有人说是寄生于语言的世界模型，有人说是通用的 Sequence to Sequence 模型，还有人说是人类文本的有损压缩，不一而足。不客气地讲，只要一个人见识够窄、读得够少、认识的人够单一、足够喜欢 hype 又足够懒得学习，那这世上总有一款“本质”能拿来说服他自己。

当然，任何一种深度的还原论本身都没错，问题在于排斥其他的还原深度，单把自己刚好看懂的那一层奉为圭臬。哪怕你不直说，这种独断其实也很容易暴露自己没什么朋友——但凡身边多几个不同背景的朋友聊聊天，也不至于不知道同一个问题原来有这么多解释角度。这个现象在大一新生身上尤其常见。每年都有新生看完 3Blue1Brown 的视频，恍然大悟，觉得旋转和缩放才是线性代数的“本质”，于是在新生群里到处看不起别人。结果一到期末考试，不少掌握了所谓线性代数本质的同学，连基本的题都算不明白；考完还要发牢骚，感慨现在的本科教育压制人才、全是应试糟粕——算行列式有什么屁用啊？行列式不就是体积吗？手算这个体积有什么现实意义啊？前后的反差，只能说多少有点幽默。

既然每个还原深度都有它的适用场景，那我不妨也用一套比较庸俗的还原论来理解下 AI 对就业的冲击。我无意宣称这是无可辩驳的本质视角，只是想借这个视角解答下我自己的困惑——AI 会把整个世界吃掉吗？倘若如此，我们又该往哪去呢？

---

如果顺着近几年的演进轨迹往下看，答案恐怕不太乐观：也许压根就不存在什么能从 AI 手里逃难的地方。

起初我们觉得，模型只能拟合一些表层的语言偏好，做做情感正负向分类；后来拿它干各类文本任务、做通用 Chatbot；再往后发现它能写代码，还能根据环境报错自己修改，于是又演进出了 Agent。刚进 Agent 时代那会儿，大家普遍认为它只适合纯自动化的闭环任务，毕竟有人参与的流程很难做强化学习，那怎么训得好呢？结果没过多久，有人参与的协作任务它也干得不错。大家只好再退一步，说数字世界里训出来的模型，总归干不好和物理现实沾边的脏活累活吧？可眼下各种复杂的 Infra 搭建、千奇百怪的现实场景适配，它照样能接手。到最后，人们搬出人类的主观直觉当最后的阵地，比如所谓的 Research taste；但最近隐约出现的一些征兆，似乎连这个故事也快讲不下去了——毕竟连千禧年七大数学难题都在逐一松动。

接下来人们肯定还会找出新的理由，去论证某类工作 AI 干不好。我现在也没有确凿的证据能断定这就是故事的终点，但从过往的经验看，分析现代社会的运转，你给它越少的附加假设、用越还原论的手段去拆解，往往越有解释力——就像当年商鞅不管六国那些复杂的宗法礼俗，直接把国家动员能力还原成“粮食产量”和“斩首数量”两项考核指标一样，虽然粗糙，但在解释秦国扩张上极其管用。

放在今天，我的理由只有两条：**资本主义的激励机制，加上神经网络的表达能力。** 只要我们假设神经网络没有天然的表达上限，Scaling Law 也没有天然的上限，那么资本主义就会自动补齐剩下的所有环节。从业者们为了维持现有的高薪，或者为了争夺更高的收入，会主动替大模型去搜寻一切尚未被攻克的业务盲区。这批人足够聪明，也能调动庞大的资金与各行各业协作，去获取数据、搭建训练环境；再加上 AI 工具本身的辅助，研发迭代的速度只会越来越快。只要某个环节还存在利润，还有人觉得“这事 AI 干不了”，资本就会出钱雇最聪明的人去把它形式化、塞进训练管线里。在这个循环下，AI 吃掉这世界的一切，大概率只是时间问题。

---

那么在这个略显悲观的假设下，我们还能干点什么呢？我想可能只有两件事，或者准确点说，其实是一件事——那就是**名声**。你不但要声名远扬，而且得有一个好名声，这样大家都认识你、信任你，你才能把自己捆绑在人类社会的分配机制上，或者更直白地说，捆绑在资本主义这辆战车上。只要这辆战车还在运转，坐在车上的你就持续有活干。

把自己捆上资本主义的战车，听起来可能不太体面，但真想在车上坐稳，靠的恰恰是最朴素的公共服务与实事求是。十六世纪宗教改革的时候，托马斯·闵采尔想脱离现实权力直接搞人间天国，很快就连人带理想一块儿失败了；马丁·路德之所以能成事，一方面是他很清醒地把自己绑上了德意志世俗诸侯的财政与政治诉求上，另一方面是他真干了一件“为人民服务”的基础工程——把拉丁文《圣经》翻译成通俗德语，大幅降低了普通人的阅读门槛，让自己成了新社会网络里绕不开的节点。放到今天的技术社区里也是一样：不论是给大家提供扎实的新知识，还是搭建好用的开源工具与协作平台，只有对社区建设产生实际帮助，把自己的工作嵌进别人的日常流程里，才谈得上所谓的声名远扬。

而要攒下“好名声”，归结起来无非两类事情：一是本身业务得**做得够牛逼**，没有过硬的产出，名声就成了无本之木；二是去干那些所有人都知道值得干、却因为现实激励扭曲而没人干成的好事。我现在想来，这种品质说白了就是四个字：**实事求是**。

眼下的行业里，大家都热衷于 overclaim，热衷于制造 hype 来争夺注意力。道理谁都懂：老老实实说话对短期宣传不利，不把故事讲大一点，怎么迅速拿成果、争预算呢？结果就是，尽管所有人都愿意和实事求是的人合作，但所有人在行动上又被迫去干虚张声势的事。正因如此，实事求是反而从道德要求变成了一种很实际的竞争手段。当年刘邦进咸阳，诸将忙着瓜分金帛财物，搁今天大概相当于趁着技术风口赶紧发通稿、刷榜单变现；唯独萧何去秦丞相府收走了律令、户籍和图书档案。夸大宣传或许有助于前期融资，可一旦组织真要往下运转，要排预算、买算力、定技术路线的时候，失真的信息就会直接拉高试错成本。谁平时积累了不注水的信用，谁能给出准确的判断，谁就能帮整个系统省钱。战车留着这样的人，不是出于道德表彰，而是因为替换他的核验成本太高。

---

好啦，说了这么多，如果再往后退一步审视，这套“捆上战车”的盘算其实也有它的边界。

古人并不比我们笨，他们只是缺少今天这套概率论与系统工程的语言工具罢了；上面推演的这些利害关系，当年的聪明人想来也不难看透。可翻开历史，无数看清大势、成功挤上时代战车的聪明人，最终却未必能落得善终，以至于后人常像白居易那样感慨，“向使当初身便死”，早点退场反而能留下一世清名。原因其实也不复杂：社会历史本身不是一个平稳系统，战车开起来之后有它自己的惯性。试图在今天就把未来几十年的安全身位一次性算清楚，本质上也是对个人预测能力的一种 overclaim——这和那个觉得背下“行列式等于体积”就能应付整门考试的大一新生，在思维方式上并没有太大区别。

回过头看，在前面提到的几条原则里，大概只有**实事求是**是跨越不同历史时期和社会组织形态都始终成立的规则，因为只要存在社会化分工与协作，降低信任成本就永远有价值。而把实事求是贯彻到个人选择上，恰恰意味着承认我们算不准终局，老老实实走一步看一步，不必提前预支那么多长远的焦虑。

回到最开始感慨的问题，大家都在想着逃离 AI，想着怎么让自己持续生活下去，这当然是有道理、完全可以理解的。但就像 Sutton 说过的，人的精力终归有限，你想要 scale 某一个属性，就很难同时去 scale 其他属性。与其在算法、Infra 和业务之间来回折腾着逃难，不如做点取舍，把精力收回到当下真正重要的事情上来。至于具体干什么能让人活得更久一点，我肯定也不知道；但我现在猜想，你眼下选择去做的事，至少应该具备刚刚说的那三个属性：**实事求是，做得够牛逼，以及为人民服务。** 先把当下的这三点做好，我想，大概也就足够了吧。

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="en" lang="en" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR:** Pivoting back and forth across algorithms, infrastructure, and business logic to outrun AI mistakes an outsider's switching costs for a durable moat. Driven by capitalist incentives and neural network expressivity, static technical barriers will likely be flattened one by one; rather than fleeing, the pragmatic play is to strap yourself onto society's resource-allocation chariot by building undeniable work, serving the commons, and practicing uncompromising intellectual honesty (_seeking truth from facts_) in a market full of hype. Yet true intellectual honesty also means admitting we cannot pre-compute a permanently safe seat in a non-stationary world—drop the endgame anxiety, take things one step at a time, and focus on doing formidable, honest, useful work right now.

</div>

People have a funny habit of enshrining whatever they don't yet understand, mistaking their own unfamiliarity for a moat. It offers a soothing excuse: a way to convince yourself that your chosen field still has headroom, that this headroom is protected by real barriers to entry, and that your livelihood won't be wiped out overnight. You see this psychology everywhere—not just in tech, but across nations, academic disciplines, and belief systems.

Still, as a survival strategy, it rarely holds up. Over the past few years, I've watched countless machine learning researchers pivot into infrastructure, while droves of systems engineers scramble to learn modeling and algorithms. Each side assumes the other holds some defensible technical barrier while their own turf is being eaten alive by AI, so they cross the aisle as refugees. And what about the people fluent in both? They already know that between automated research loops tuning algorithms and automated compilers optimizing kernels, neither side has much of a wall left. So they pin their hopes on messy, real-world "business logic," praying that proximity to human friction will shield them from AI. Every group has a neat argument for why their new shelter is **"fundamentally different"** from the territory they just abandoned.

Issuing yourself an immunity pass by declaring your new niche "fundamentally different" isn't an invention of modern software engineers. When the Qin war machine rolled east in the third century BCE, the literati of the six rival states almost certainly believed the ritual traditions of Qi and Lu were "fundamentally different" from Qin's agrarian-military arithmetic. When Gutenberg's movable type began to spread, medieval scholastics surely trusted that Latin grammar and Aristotelian commentary possessed an unbreachable barrier of their own. People routinely confuse the switching cost an outsider faces when learning a craft with an objective wall that technology cannot flatten.

---

"Fundamentally different" is a pretty comical phrase when you think about it—because who gets to decide what the "fundamental essence" of anything actually is? An "essence" is determined entirely by how many layers of reductionism you choose to peel back. Stop wherever you like, and that layer becomes your essence. Take large language models: one person insists they are essentially probability distributions; another says they are world models parasitic on human language; a third calls them general-purpose sequence-to-sequence mappers; a fourth describes them as lossy compressions of human text. To put it bluntly, as long as your exposure is narrow enough, your reading list short enough, your social circle homogeneous enough, and your appetite for hype much stronger than your appetite for study, there will always be a ready-made "essence" waiting to convince you.

There is nothing wrong with reductionism at any given depth. The trouble starts when you treat the single layer you happen to grasp as gospel and dismiss every other level of explanation. Even if you never say it out loud, that kind of dogmatism quietly gives away that you don't have many friends—otherwise, basic conversations with people from different backgrounds would have shown you how many valid cross-sections the same problem can have. You see this constantly among college freshmen. Every fall, a fresh batch of first-years watches 3Blue1Brown, has an epiphany that rotation and scaling are the "true essence" of linear algebra, and spends the semester looking down on everyone else in the group chat. Then finals arrive, and plenty of these students who have grasped the "essence" of linear algebra can't work through a basic problem. Afterward, they vent about how undergraduate education stifles talent with rote exam prep—_what's the point of computing determinants by hand? Isn't a determinant just volume? What practical meaning does calculating that volume even have?_ Watching the whole sequence play out is, at the very least, mildly entertaining.

Since every level of reductionism has its uses, let me borrow a fairly vulgar reductionism of my own to think through what AI means for human work. I make no claim that this is the ultimate, irrefutable lens—I just want to use it to sort through a nagging question of my own: Is AI going to eat the entire world? And if it does, where are we supposed to go?

---

If you extrapolate from the trajectory of the past few years, the outlook isn't terribly comforting. There may not be anywhere left to flee.

At first, we assumed neural networks could only capture surface-level linguistic preferences, like classifying sentiment as positive or negative. Then we put them to work on general text tasks and built chatbots. Soon we noticed they could write code and patch bugs from environment tracebacks, which gave rise to agents. Early in the agent era, the consensus was that they would only excel at clean, closed-loop automation—after all, workflows involving humans in the loop are notoriously messy to optimize with reinforcement learning, so how could you possibly train them well? Before long, they were handling human-in-the-loop workflows quite capably too. So people retreated another step: fine, a model trained inside digital sandboxes surely won't handle the grimy, ad-hoc realities of physical infrastructure and edge-case deployment. Yet today, models are standing up complex clusters and adapting to bizarre operational environments just fine. Finally, people fell back on subjective human intuition—things like "research taste"—as the last bastion. Lately, though, faint tremors suggest even that story may be nearing its end, as one Millennium Prize Problem after another begins to show cracks.

People will undoubtedly invent fresh reasons why the next category of work is beyond AI's reach. I don't have hard proof that the story ends in total automation either. Still, history suggests that when you analyze how modern societies operate, stripping away extra assumptions and applying cruder reductionism often yields stronger explanatory power—much like how the Qin reformer Shang Yang ignored the intricate clan etiquette of rival states and reduced state mobilization to two scalable metrics: grain output and enemy heads collected. Crude as that formula was, it explained Qin's expansion remarkably well.

Today, my reductionist premise comes down to two forces: **capitalist incentives and the expressive capacity of neural networks.** If we assume neither neural network expressivity nor scaling laws face an imminent natural ceiling, capitalism will supply every missing piece on its own. Practitioners trying to defend their current compensation—or capture an even larger share—will actively hunt down every unautomated pocket of the economy on the model's behalf. These people are remarkably smart, capable of mobilizing vast capital and partnering across industries to acquire proprietary data and build training environments, while AI tools themselves accelerate every cycle of R&D. Wherever profit remains and someone still claims "AI can't do this," capital will pay the brightest minds on earth to formalize the task and feed it into the training pipeline. Under that feedback loop, AI eating everything is likely just a matter of time.

---

Under such a bleak assumption, what is left for us to do? Two things, perhaps—or more accurately, two halves of a single thing: **reputation**. Not only do you need to be widely known, you need to be trusted. Only then can you strap yourself onto society's resource-allocation machinery—or, to put it less politely, **strap yourself onto the chariot of capitalism**. As long as that chariot keeps rolling, anyone bolted onto its frame will have work to do.

Strapping yourself onto the chariot of capitalism sounds mercenary, yet staying seated on it actually depends on two remarkably plain habits: public service and intellectual honesty. During the sixteenth-century Reformation, Thomas Müntzer tried to bypass secular power altogether to build Heaven on Earth, and both he and his movement were swiftly crushed. Martin Luther survived and reshaped Europe partly because he pragmatically hitched his cause to the fiscal and political ambitions of German secular princes—but his leverage came from doing a massive piece of public infrastructure first: translating the Latin Bible into plain vernacular German. Lowering the barrier to literacy embedded him directly into the dependency graph of the emerging social order. The same logic holds in technical communities today. Whether you contribute solid new knowledge or maintain open tools and shared platforms, you only become widely known in a durable way when your work genuinely serves the commons and sits inside other people's daily workflows.

Earning a _good_ reputation, meanwhile, boils down to two requirements. First, your work has to be **undeniably strong**; without real technical teeth, reputation has nothing to stand on. Second, you have to do the things everyone knows are worthwhile, yet nobody finishes because short-term incentives are warped. Put plainly, that quality is what we call _shishi qiushi_—**seeking truth from facts**, or uncompromising intellectual honesty.

Right now, the industry rewards overclaiming and manufacturing hype to capture attention. Everyone understands the game: candor hurts short-term promotion, and if you don't inflate the story, how do you secure fast results, headcount, or funding? Consequently, while everyone prefers to collaborate with honest people in private, almost everyone feels pressured to bluff in public. That exact distortion turns intellectual honesty from a moral platitude into a practical competitive edge. When Liu Bang's rebel army first marched into the Qin capital of Xianyang, his generals rushed into the treasuries to divide up gold and silk—the ancient equivalent of cashing in on a hype cycle with flashy press releases and cherry-picked benchmarks. Xiao He alone walked over to the chancellor's archive and packed up the legal statutes, census registers, and geographic ledgers. Inflated narratives may help raise an initial round, but once an organization actually has to operate—allocating budgets, buying compute, committing to technical roadmaps—noisy, distorted signals drive up the cost of mistakes. Whoever has accumulated a track record of unvarnished judgment saves the system real money. The chariot keeps people like that on board not as a reward for virtue, but because the verification cost of replacing them is too high.

---

Even so, if we step back one layer further, this whole calculus of "strapping onto the chariot" has limits of its own.

People in antiquity were no less intelligent than we are; they simply lacked the vocabulary of probability theory and systems engineering. The strategic trade-offs sketched above would hardly have escaped perceptive minds back then. Yet history is littered with brilliant operators who read the tide correctly, climbed aboard the winning chariot of their era, and still met grim ends—prompting the Tang poet Bai Juyi's famous sigh that had figures like Wang Mang died early, history would remember them as saints rather than usurpers. The reason isn't mysterious: history is a non-stationary system, and once a juggernaut picks up speed, its own momentum can easily devour its passengers. Trying to compute a permanently safe position decades ahead of time is itself an overclaim about one's predictive capacity—not all that different, in spirit, from the freshman convinced that memorizing "determinants equal volume" is enough to ace the exam.

Looking back over everything discussed here, **seeking truth from facts** is probably the only rule that holds up across eras and forms of social organization—because wherever human division of labor exists, lowering the cost of trust retains its value. Applied to one's own life, intellectual honesty begins with admitting that we cannot calculate the endgame, taking things one step at a time rather than pre-borrowing decades of existential anxiety.

To return to where we started: everyone is looking for an escape route from AI, trying to figure out how to stay afloat, and that impulse is completely understandable. Still, as Rich Sutton once noted, human attention is finite—if you choose to scale one dimension, you inevitably sacrifice your ability to scale others. Rather than burning energy migrating back and forth across algorithms, infrastructure, and business logic like a perpetual refugee, it makes more sense to make a clean cut and focus on what matters right in front of us. What exact line of work will let someone survive the longest, I certainly don't know. My best guess is simply that whatever you choose to work on right now ought to carry three traits: **it seeks truth from facts, it is genuinely formidable work, and it serves the community.** Get those three right in the present, and I suspect that will be enough.

</div>

<script>
(function () {
  var dispatch = document.querySelector(".zine-dispatch");
  var menuList = document.querySelector("#navbar .navbar-menu-list");
  if (menuList && dispatch && !menuList.querySelector(".zine-nav-lang")) {
    var navItem = document.createElement("li");
    navItem.className = "nav-item zine-nav-lang";
    var navSwitch = document.createElement("div");
    navSwitch.className = "zine-lang-switch";
    navSwitch.setAttribute("role", "group");
    navSwitch.setAttribute("aria-label", "Language");
    var btnEn = document.createElement("button");
    btnEn.type = "button";
    btnEn.className = "zine-lang-btn";
    btnEn.setAttribute("data-lang-target", "en");
    btnEn.setAttribute("aria-pressed", "false");
    btnEn.textContent = "EN";
    var btnZh = document.createElement("button");
    btnZh.type = "button";
    btnZh.className = "zine-lang-btn is-active";
    btnZh.setAttribute("data-lang-target", "zh");
    btnZh.setAttribute("lang", "zh-CN");
    btnZh.setAttribute("aria-pressed", "true");
    btnZh.textContent = "中文";
    navSwitch.appendChild(btnEn);
    navSwitch.appendChild(btnZh);
    navItem.appendChild(navSwitch);
    var searchBtn = document.getElementById("search-toggle");
    var searchLi = searchBtn ? searchBtn.closest("li") : null;
    if (searchLi && searchLi.parentNode === menuList) {
      menuList.insertBefore(navItem, searchLi);
    } else {
      menuList.appendChild(navItem);
    }
  }
  var buttons = document.querySelectorAll("[data-lang-target]");
  var panels = document.querySelectorAll("[data-lang-panel]");
  var heading = document.querySelector(".post-header h1.post-title");
  var i18nNodes = document.querySelectorAll("[data-i18n-zh][data-i18n-en]");
  var titleSep = document.title.indexOf(" | ");
  var titleSuffix = titleSep !== -1 ? document.title.slice(titleSep) : "";
  if (!buttons.length || !panels.length) return;
  function applyLang(lang, updateUrl) {
    var target = lang === "en" ? "en" : "zh";
    buttons.forEach(function (btn) {
      var active = btn.getAttribute("data-lang-target") === target;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-pressed", active ? "true" : "false");
    });
    panels.forEach(function (panel) {
      var match = panel.getAttribute("data-lang-panel") === target;
      panel.classList.toggle("is-hidden", !match);
    });
    if (dispatch) {
      var nextTitle = dispatch.getAttribute(target === "en" ? "data-title-en" : "data-title-zh");
      if (nextTitle) {
        if (heading) heading.textContent = nextTitle;
        document.title = nextTitle + titleSuffix;
      }
    }
    i18nNodes.forEach(function (el) {
      el.textContent = el.getAttribute(target === "en" ? "data-i18n-en" : "data-i18n-zh");
    });
    document.documentElement.lang = target === "en" ? "en" : "zh-CN";
    document.documentElement.classList.toggle("lang-zh", target === "zh");
    document.documentElement.classList.toggle("lang-en", target === "en");
    if (updateUrl) {
      try {
        var url = new URL(window.location.href);
        url.searchParams.set("lang", target);
        window.history.replaceState(null, "", url.toString());
      } catch (e) {}
    }
  }
  buttons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      applyLang(btn.getAttribute("data-lang-target"), true);
    });
  });
  var initial = "zh";
  try {
    var param = new URLSearchParams(window.location.search).get("lang");
    if (param === "en" || param === "zh") initial = param;
  } catch (e) {}
  applyLang(initial, false);
})();
</script>
