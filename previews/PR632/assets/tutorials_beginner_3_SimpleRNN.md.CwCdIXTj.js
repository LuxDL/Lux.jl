import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.DlIaf7OG.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="Dataset" tabindex="-1">Dataset <a class="header-anchor" href="#Dataset" aria-label="Permalink to &quot;Dataset {#Dataset}&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR632/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR632/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR632/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR632/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    result </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ifelse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">iszero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">zero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(result), result)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> binarycrossentropy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> eps</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@.</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_true, y_pred) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_pred))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> binarycrossentropy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y), st, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_pred)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            gs, loss, _, train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">compute_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), compute_loss, (x, y), train_state)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">apply_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state, gs)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss, st_, ret </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ret</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_pred, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56171</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50756</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49201</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45849</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42627</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41174</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40106</span></span>
<span class="line"><span>Validation: Loss 0.36743 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37688 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37975</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35958</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33529</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31762</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30398</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29257</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28185</span></span>
<span class="line"><span>Validation: Loss 0.25958 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26507 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26260</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25716</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23708</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22697</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21437</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20246</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18485</span></span>
<span class="line"><span>Validation: Loss 0.18390 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18641 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18462</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17637</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16880</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16343</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15476</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14645</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13641</span></span>
<span class="line"><span>Validation: Loss 0.13344 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13535 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13505</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12745</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12324</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11976</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11225</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10929</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10147</span></span>
<span class="line"><span>Validation: Loss 0.09839 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10035 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09993</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09506</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09138</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08835</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08472</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08007</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07923</span></span>
<span class="line"><span>Validation: Loss 0.07331 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07536 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07511</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07337</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06733</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06589</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06571</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05886</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05556</span></span>
<span class="line"><span>Validation: Loss 0.05502 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05702 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05528</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05329</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04973</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05019</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04857</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04881</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04533</span></span>
<span class="line"><span>Validation: Loss 0.04138 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04327 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04236</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04295</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03875</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03689</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03590</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03473</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03539</span></span>
<span class="line"><span>Validation: Loss 0.03139 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03307 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03141</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03120</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03158</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02949</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02821</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02745</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02194</span></span>
<span class="line"><span>Validation: Loss 0.02448 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02595 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02631</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02463</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02490</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02240</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02271</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02096</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01931</span></span>
<span class="line"><span>Validation: Loss 0.01981 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02110 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02074</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02049</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02006</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01922</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01868</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01734</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01600</span></span>
<span class="line"><span>Validation: Loss 0.01656 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01769 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01822</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01558</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01560</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01584</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01595</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01689</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01502</span></span>
<span class="line"><span>Validation: Loss 0.01423 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01523 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01530</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01403</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01448</span></span>
<span class="line"><span>Validation: Loss 0.01248 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01337 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01308</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01233</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01223</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01187</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01179</span></span>
<span class="line"><span>Validation: Loss 0.01111 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01191 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01166</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01117</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00956</span></span>
<span class="line"><span>Validation: Loss 0.01001 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01073 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01016</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01054</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01031</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00947</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01055</span></span>
<span class="line"><span>Validation: Loss 0.00910 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00977 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00950</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00901</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00964</span></span>
<span class="line"><span>Validation: Loss 0.00833 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00895 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00880</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00841</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00836</span></span>
<span class="line"><span>Validation: Loss 0.00767 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00824 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00780</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00737</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00749</span></span>
<span class="line"><span>Validation: Loss 0.00709 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00763 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00703</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00743</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00729</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00714</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00671</span></span>
<span class="line"><span>Validation: Loss 0.00659 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00709 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00699</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00624</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00716</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00695</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00666</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00642</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00655</span></span>
<span class="line"><span>Validation: Loss 0.00615 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00662 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00619</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00618</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00657</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00625</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00649</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00690</span></span>
<span class="line"><span>Validation: Loss 0.00576 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00621 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00597</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00631</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00606</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00565</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00586</span></span>
<span class="line"><span>Validation: Loss 0.00541 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00583 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00541</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00569</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00550</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00464</span></span>
<span class="line"><span>Validation: Loss 0.00509 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00549 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56277</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51155</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48436</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45007</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42725</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39463</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37197</span></span>
<span class="line"><span>Validation: Loss 0.37211 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36207 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36636</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35411</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33374</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32448</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30076</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28223</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26928</span></span>
<span class="line"><span>Validation: Loss 0.26132 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25409 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25869</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24533</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22921</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21840</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21531</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19955</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19326</span></span>
<span class="line"><span>Validation: Loss 0.18176 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17765 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17974</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17423</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15931</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15418</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14794</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14265</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13602</span></span>
<span class="line"><span>Validation: Loss 0.12912 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12663 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12683</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12143</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11574</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11430</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10803</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10148</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09402</span></span>
<span class="line"><span>Validation: Loss 0.09386 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09173 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09144</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09019</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08610</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07947</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07785</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07631</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07273</span></span>
<span class="line"><span>Validation: Loss 0.06944 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06728 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06928</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06488</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06171</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05892</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05778</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05769</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05894</span></span>
<span class="line"><span>Validation: Loss 0.05201 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04993 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05157</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04997</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04803</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04565</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04257</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04132</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03753</span></span>
<span class="line"><span>Validation: Loss 0.03928 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03744 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03825</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03660</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03653</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03449</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03279</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03196</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03076</span></span>
<span class="line"><span>Validation: Loss 0.03016 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02845 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02968</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02821</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02782</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02732</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02550</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02406</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02400</span></span>
<span class="line"><span>Validation: Loss 0.02379 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02226 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02138</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02280</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02160</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02135</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02101</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02091</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02046</span></span>
<span class="line"><span>Validation: Loss 0.01942 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01948</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01884</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01889</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01669</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01644</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01538</span></span>
<span class="line"><span>Validation: Loss 0.01631 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01515 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01632</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01528</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01582</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01451</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01291</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01557</span></span>
<span class="line"><span>Validation: Loss 0.01410 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01308 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01375</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01317</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01319</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01247</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01170</span></span>
<span class="line"><span>Validation: Loss 0.01243 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01152 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01204</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01158</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01012</span></span>
<span class="line"><span>Validation: Loss 0.01115 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01032 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00994</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01063</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01029</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01086</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01137</span></span>
<span class="line"><span>Validation: Loss 0.01009 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00933 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00913</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00950</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00943</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00919</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00907</span></span>
<span class="line"><span>Validation: Loss 0.00922 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00851 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00928</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00811</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00810</span></span>
<span class="line"><span>Validation: Loss 0.00847 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00782 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00770</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00787</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00764</span></span>
<span class="line"><span>Validation: Loss 0.00783 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00722 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00740</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00750</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00744</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00696</span></span>
<span class="line"><span>Validation: Loss 0.00728 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00670 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00750</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00716</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00647</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00687</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00649</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00667</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00767</span></span>
<span class="line"><span>Validation: Loss 0.00679 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00625 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00631</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00617</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00648</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00642</span></span>
<span class="line"><span>Validation: Loss 0.00635 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00584 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00618</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00637</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00568</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00596</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00638</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00518</span></span>
<span class="line"><span>Validation: Loss 0.00596 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00548 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00557</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00564</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00558</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00537</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00620</span></span>
<span class="line"><span>Validation: Loss 0.00561 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00516 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00508</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00573</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00511</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00511</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00547</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00531</span></span>
<span class="line"><span>Validation: Loss 0.00529 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00487 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxAMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Julia Version 1.10.3</span></span>
<span class="line"><span>Commit 0b4590a5507 (2024-04-30 10:59 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 8 default, 0 interactive, 4 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_AMDGPU_LOGGING_ENABLED = true</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 8</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.4, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.4</span></span>
<span class="line"><span>NVIDIA driver 550.54.15</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.4.5</span></span>
<span class="line"><span>- CURAND: 10.3.5</span></span>
<span class="line"><span>- CUFFT: 11.2.1</span></span>
<span class="line"><span>- CUSOLVER: 11.6.1</span></span>
<span class="line"><span>- CUSPARSE: 12.3.1</span></span>
<span class="line"><span>- CUPTI: 22.0.0</span></span>
<span class="line"><span>- NVML: 12.0.0+550.54.15</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.3.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.8.1+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.12.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.3</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.391 GiB / 4.750 GiB available)</span></span>
<span class="line"><span>┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.</span></span>
<span class="line"><span>└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,k,c,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
