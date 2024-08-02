import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.CmpXK8-u.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR801/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR801/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR801/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR801/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56304</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51211</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46957</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45357</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42503</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40882</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38250</span></span>
<span class="line"><span>Validation: Loss 0.36518 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36511 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36539</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34566</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33686</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31572</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30340</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28844</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27382</span></span>
<span class="line"><span>Validation: Loss 0.25554 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25572 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25832</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24051</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22816</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22222</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20992</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20036</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18897</span></span>
<span class="line"><span>Validation: Loss 0.17809 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17829 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17902</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17077</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16089</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15476</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14624</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13986</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13444</span></span>
<span class="line"><span>Validation: Loss 0.12664 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12676 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12597</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12103</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11432</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11267</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10658</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09958</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09880</span></span>
<span class="line"><span>Validation: Loss 0.09165 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09167 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09170</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08797</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08443</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08025</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07706</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07564</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06711</span></span>
<span class="line"><span>Validation: Loss 0.06732 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06726 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06760</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06593</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06300</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05775</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05802</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05474</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05236</span></span>
<span class="line"><span>Validation: Loss 0.05002 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04991 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04877</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04983</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04760</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04457</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04282</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04108</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03947</span></span>
<span class="line"><span>Validation: Loss 0.03754 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03743 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03730</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03641</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03668</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03407</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03231</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03137</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02883</span></span>
<span class="line"><span>Validation: Loss 0.02860 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02850 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02918</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02930</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02584</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02667</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02618</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02343</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02293</span></span>
<span class="line"><span>Validation: Loss 0.02245 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02237 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02274</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02239</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02141</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02122</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01959</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01977</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02166</span></span>
<span class="line"><span>Validation: Loss 0.01825 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01819 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01847</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01841</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01850</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01739</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01613</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01655</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01508</span></span>
<span class="line"><span>Validation: Loss 0.01531 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01527 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01512</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01456</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01442</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01466</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01541</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01268</span></span>
<span class="line"><span>Validation: Loss 0.01322 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01318 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01323</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01269</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01279</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01293</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01275</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01407</span></span>
<span class="line"><span>Validation: Loss 0.01165 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01162 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01144</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01191</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01059</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01090</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01110</span></span>
<span class="line"><span>Validation: Loss 0.01042 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01039 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01125</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01047</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01006</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01017</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00983</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00923</span></span>
<span class="line"><span>Validation: Loss 0.00943 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00941 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00988</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00866</span></span>
<span class="line"><span>Validation: Loss 0.00861 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00858 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00922</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00878</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00842</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00886</span></span>
<span class="line"><span>Validation: Loss 0.00791 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00789 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00753</span></span>
<span class="line"><span>Validation: Loss 0.00730 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00729 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00732</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00714</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00728</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00726</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00716</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00642</span></span>
<span class="line"><span>Validation: Loss 0.00678 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00677 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00658</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00723</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00663</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00681</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00675</span></span>
<span class="line"><span>Validation: Loss 0.00632 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00631 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00635</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00661</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00657</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00616</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00524</span></span>
<span class="line"><span>Validation: Loss 0.00592 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00590 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00616</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00617</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00609</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00588</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00557</span></span>
<span class="line"><span>Validation: Loss 0.00555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00554 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00641</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00552</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00569</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00569</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00528</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00509</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00492</span></span>
<span class="line"><span>Validation: Loss 0.00523 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00521 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00543</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00535</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00513</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00535</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00526</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00500</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00539</span></span>
<span class="line"><span>Validation: Loss 0.00493 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00492 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56323</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51059</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47735</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45075</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41810</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40747</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37464</span></span>
<span class="line"><span>Validation: Loss 0.36294 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37340 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37406</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35357</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32519</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31585</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29671</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29081</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26640</span></span>
<span class="line"><span>Validation: Loss 0.25437 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26035 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26025</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24506</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23353</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21929</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20807</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19631</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18953</span></span>
<span class="line"><span>Validation: Loss 0.17739 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18006 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17832</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17258</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16143</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15545</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14664</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13869</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13198</span></span>
<span class="line"><span>Validation: Loss 0.12608 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12755 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12671</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12159</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11648</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11112</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10411</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10172</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09372</span></span>
<span class="line"><span>Validation: Loss 0.09111 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09263 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09201</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08848</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08216</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08115</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07745</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07461</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07161</span></span>
<span class="line"><span>Validation: Loss 0.06679 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06858 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06965</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06350</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06119</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06068</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05733</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05473</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05180</span></span>
<span class="line"><span>Validation: Loss 0.04959 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05141 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05063</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04763</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04698</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04353</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04303</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04185</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04185</span></span>
<span class="line"><span>Validation: Loss 0.03715 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03894 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03931</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03536</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03508</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03344</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03216</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03156</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03144</span></span>
<span class="line"><span>Validation: Loss 0.02826 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02995 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02853</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02833</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02782</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02528</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02473</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02515</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02321</span></span>
<span class="line"><span>Validation: Loss 0.02216 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02368 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02445</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02211</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02028</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02193</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01961</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01892</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01914</span></span>
<span class="line"><span>Validation: Loss 0.01801 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01933 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01851</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01806</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01778</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01766</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01632</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01637</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01645</span></span>
<span class="line"><span>Validation: Loss 0.01513 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01628 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01544</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01514</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01436</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01489</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01437</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01337</span></span>
<span class="line"><span>Validation: Loss 0.01308 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01408 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01414</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01272</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01253</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01200</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01315</span></span>
<span class="line"><span>Validation: Loss 0.01153 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01242 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01256</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01134</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01163</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01086</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01201</span></span>
<span class="line"><span>Validation: Loss 0.01031 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01112 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01023</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00901</span></span>
<span class="line"><span>Validation: Loss 0.00933 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01006 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00927</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00869</span></span>
<span class="line"><span>Validation: Loss 0.00851 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00919 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00790</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00841</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00936</span></span>
<span class="line"><span>Validation: Loss 0.00782 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00846 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00821</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00753</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00694</span></span>
<span class="line"><span>Validation: Loss 0.00723 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00781 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00780</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00727</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00745</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00665</span></span>
<span class="line"><span>Validation: Loss 0.00671 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00726 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00696</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00717</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00647</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00649</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00705</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00600</span></span>
<span class="line"><span>Validation: Loss 0.00625 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00677 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00627</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00613</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00629</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00638</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00637</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00652</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00643</span></span>
<span class="line"><span>Validation: Loss 0.00585 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00634 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00617</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00610</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00642</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00612</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00513</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00587</span></span>
<span class="line"><span>Validation: Loss 0.00549 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00595 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00578</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00506</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00558</span></span>
<span class="line"><span>Validation: Loss 0.00517 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00560 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00554</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00533</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00515</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00470</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00485</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00571</span></span>
<span class="line"><span>Validation: Loss 0.00487 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00528 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxDeviceUtils)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.4</span></span>
<span class="line"><span>Commit 48d4fd48430 (2024-06-04 10:41 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 4 default, 0 interactive, 2 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 4</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.3</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.3</span></span>
<span class="line"><span>- CUSPARSE: 12.5.1</span></span>
<span class="line"><span>- CUPTI: 2024.2.1 (API 23.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.1+1</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.4</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
